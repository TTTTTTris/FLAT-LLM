import time
import logging
import torch
import torch.nn as nn

from lib.data_utils import get_dataset, prepare_dataloader, prepare_test_dataloader
from lib.eval import eval_ppl, eval_zero_shot
from lib.prune import prepare_calibration_input, find_layers
from transformers import AutoTokenizer, AutoModelForCausalLM
from gptq.gptq import *
from gptq.quant import *

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    print("model sequence length: ", model.seqlen)
    return model

from lib.svd_llm import CustomLlamaDecoderLayer

def replace_decoder_layers(model):
    """
    Replace all decoder layers in a LLaMA model with CustomLlamaDecoderLayer,
    preserving original weights and layer indices.
    """

    for i, org_dec in enumerate(model.model.layers):
        org_dec.to('cpu')
        torch.cuda.empty_cache()
        with torch.no_grad():
            # Instantiate custom decoder with layer index
            new_dec = CustomLlamaDecoderLayer(model.config, layer_idx=i)
            
            # Load weights from original decoder
            new_dec.load_state_dict(org_dec.state_dict(), strict=True)

            # Move to same device and dtype as original
            new_dec.to(device=next(org_dec.parameters()).device,
                       dtype=next(org_dec.parameters()).dtype)

            # Replace in model
            model.model.layers[i] = new_dec

            print(f"Replaced decoder layer {i + 1}/{len(model.model.layers)}")

        # Free memory
        del org_dec
        torch.cuda.empty_cache()

    return model

from functools import reduce

def get_nested_attr(obj, attr_path):
    """Access nested attribute like 'encoder.layer.0.mlp.fc1'."""
    return reduce(getattr, attr_path.split('.'), obj)

def set_nested_attr(obj, attr_path, value):
    """Set nested attribute like 'encoder.layer.0.mlp.fc1'."""
    parts = attr_path.split('.')
    parent = reduce(getattr, parts[:-1], obj)
    setattr(parent, parts[-1], value)

@torch.no_grad()
def resize_and_substitute_linear_layers(model, state_dict):
    updated_keys = set()

    for key in state_dict:
        if not key.endswith(".weight"):
            continue

        base_key = key[:-7]  # remove '.weight'
        weight = state_dict[key]

        try:
            layer = get_nested_attr(model, base_key)
        except AttributeError:
            continue  # Skip missing layers

        if not isinstance(layer, nn.Linear):
            continue  # Only process nn.Linear layers

        # Check if shape mismatch
        if layer.weight.shape != weight.shape:
            print(f"[RESIZE] {base_key}: {layer.weight.shape} → {weight.shape}")

            # Move old layer to CPU to free GPU memory early
            layer.to("cpu")
            torch.cuda.empty_cache()

            in_features = weight.shape[1]
            out_features = weight.shape[0]
            bias = getattr(layer, "bias", None) is not None

            # Create new linear layer and replace
            new_layer = nn.Linear(in_features, out_features, bias=bias).to(dtype=weight.dtype, device=weight.device)
            set_nested_attr(model, base_key, new_layer)

        # Now copy weights (and bias if available)
        get_nested_attr(model, base_key).weight.copy_(state_dict[key])
        updated_keys.add(key)

        bias_key = f"{base_key}.bias"
        if bias_key in state_dict:
            get_nested_attr(model, base_key).bias.copy_(state_dict[bias_key])
            updated_keys.add(bias_key)

    print(f"Loaded and resized {len(updated_keys)//2} linear layers.")
    torch.cuda.empty_cache()

@torch.no_grad()
def quantize_flatllm(args, model, tokenizer, device=torch.device("cuda:0")):
    print('Starting ...')

    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    # dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    dataset = get_dataset(args.dataset)
    train_dataset, _ = dataset["train"], dataset["test"]
    dataloader = prepare_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_seqlen=model.seqlen,
        batch_size=1,
        nsamples=args.nsamples,
        seed=args.seed,
    )

    print("dataset loading complete")
    with torch.no_grad(): # inps = data; outs = 0
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, args.nsamples, dataloader, device)

    position_ids = torch.arange(model.seqlen, dtype=torch.long, device=device).unsqueeze(0).expand(1, -1)
    layers = model.model.layers.to('cpu')

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i]
        full = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
        else:
            dev = device
        layer = layer.to(dev)
        if inps is not None:
            inps = inps.to(dev)
        if outs is not None:
            outs = outs.to(dev)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)
        if position_ids is not None:
            position_ids = position_ids.to(dev)

        # if args.true_sequential:
        #     sequential = [
        #         ['self_attn.k_u_proj','self_attn.k_v_proj', 'self_attn.v_u_proj', 'self_attn.v_v_proj', 'self_attn.q_u_proj', 'self_attn.q_v_proj'],
        #         ['self_attn.o_u_proj', 'self_attn.o_v_proj'],
        #         ['mlp.up_u_proj', 'mlp.up_v_proj', 'mlp.gate_u_proj', 'mlp.gate_v_proj'],
        #         ['mlp.down_u_proj', 'mlp.down_v_proj']
        #     ]
        # else:
        sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, 
        help='LLaMA model'
    )
    parser.add_argument(
        '--model_path', type=str, default=None,
        help='path of the compressed model.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'alpaca'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=256,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default=None, 
        help='Path to save results.'
    )
    parser.add_argument(
        '--save_model', type=str, default=None, 
        help='Path to save the pruned model.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        '--sparsity_ratio', type=int, default=0, 
        help='Sparsity level'
    )
    parser.add_argument(
        "--cache_dir", default="llm_weights", type=str
    )

    args = parser.parse_args()
    device = torch.device("cuda:0")

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Set up logging
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_gptq_{args.sparsity_ratio}.txt")
    logging.basicConfig(filename=save_filepath, level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example of logging
    logging.info("Log file created and logging started")

    logging.info(f"loading llm model {args.model}")
    
    model = get_llm(args.model, args.cache_dir)

    if args.model_path:
        # model = torch.load(args.model_path, weights_only=False)
        state_dict= torch.load(args.model_path, weights_only=False, map_location="cpu")
        if isinstance(state_dict, dict):
            model = replace_decoder_layers(model)             
            resize_and_substitute_linear_layers(model, state_dict)
            # Load the full state dict without shape checking
            model.load_state_dict(state_dict, strict=False)
        else:
            model = state_dict.to(device)
        print(f"Loaded model from {args.model_path}")

    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    if "70b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        for module_name, dev in model.hf_device_map.items():
            submodule = dict(model.named_modules())[module_name]
            dev = 'cuda:' + str(dev)
            submodule.to(dev)
    else:
        model = model.to(device)

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        # quantizers = llama_sequential(model, dataloader, args.DEV)
        quantizers = quantize_flatllm(args, model, tokenizer, device=torch.device("cuda:0"))
        print(time.time() - tick)

    if "70b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        for module_name, dev in model.hf_device_map.items():
            submodule = dict(model.named_modules())[module_name]
            dev = 'cuda:' + str(dev)
            submodule.to(dev)
    else:
        model = model.to(device)

    model.config.num_hidden_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = i

    ppl_test = eval_ppl(args, model, tokenizer, device=device)
    logging.info(f"pre-trained model {args.model} before pruning")
    logging.info(f"wikitext perplexity {ppl_test}")

    torch.save({
                'model': model,
                'tokenizer': tokenizer
            }, args.save_model) 
