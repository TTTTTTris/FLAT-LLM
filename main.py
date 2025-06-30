import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import logging
from lib.prune import prune_flatllm, check_structual_pruning, compute_bi
from lib.eval import eval_ppl, eval_zero_shot
from lib.svd_llm import CustomLlamaDecoderLayer

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = 4096
    print("model sequence length: ", model.seqlen)
    return model

def replace_decoder_layers(model, args, device):
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

def check_model_devices(model):
    if not hasattr(model, "hf_device_map"):
        print("The model does not have an hf_device_map.")
        return

    # Get the hf_device_map
    device_map = model.hf_device_map
    print("hf_device_map:")
    for module_name, device in device_map.items():
        print(f"  {module_name}: {device}")

    # Check actual devices of model components
    print("\nChecking actual devices of model components:")
    for module_name, module in model.named_modules():
        if module_name in device_map:
            # Get the expected device from hf_device_map
            expected_device = device_map[module_name]

            # Get the actual device of the module
            if len(list(module.parameters())) > 0:
                actual_device = next(module.parameters()).device
            else:
                actual_device = "No parameters"

            # Compare devices
            status = "MATCH" if str(actual_device) == expected_device else "MISMATCH"
            print(f"{module_name}: expected={expected_device}, actual={actual_device} [{status}]")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=int, default=0, help='Sparsity level')
    parser.add_argument("--prune_method", type=str, choices=["flatllm", "bi"], default=None)
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")

    # tensor args
    parser.add_argument("--tol", type=float, default=0.96, help="pruning tolerance")
    parser.add_argument("--save_sparse_ratio", action="store_true", help="Enable sparsity ratio saving")
    parser.add_argument("--ratio", type=str, default=None, help="tensor rank reduction")
    parser.add_argument('--bi_score', type=str, default=None, help='bi score')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='dataset for calibratoin', choices=['wikitext2', 'c4', 'alpaca'])
    parser.add_argument("--ft_model_name_or_path", type=str, default="", help="Path or name of the fine-tuned model")

    # Parse arguments
    args = parser.parse_args()
    device = torch.device("cuda:0")

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Create logfile
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.sparsity_ratio}.txt")
    
    # Set up logging
    logging.basicConfig(filename=save_filepath, level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example of logging
    logging.info("Log file created and logging started")

    logging.info(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # check_model_devices(model)
    # print(device)
    # ppl_test = eval_ppl(args, model, tokenizer, device=device)

    if args.prune_method != "bi":
        model = replace_decoder_layers(model, args, device)    

    model.eval()
    logging.info(model)
    
    if args.ft_model_name_or_path:
        from safetensors.torch import load_file
        import glob
        logging.info(f"loading tensor weights from {args.ft_model_name_or_path}")
        safetensor_files = sorted(glob.glob(args.ft_model_name_or_path + '/*.safetensors'))  # Auto-detect all safetensors
        combined_state_dict = {}
        # Load each safetensor file and merge into combined_state_dict
        print(safetensor_files)
        for path in safetensor_files:
            print(path)
            print(f"Loading weights from {path}...")
            state_dict = load_file(path)
            combined_state_dict.update(state_dict)  # Merge weights
        # Load the merged state_dict into the model
        model.load_state_dict(combined_state_dict, strict=False)  # strict=False ignores missing keys

    if "70b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    logging.info(f"use device {device}")
    
    # Debugging
    if "70b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        for module_name, dev in model.hf_device_map.items():
            submodule = dict(model.named_modules())[module_name]
            dev = 'cuda:' + str(dev)
            submodule.to(dev)
    elif args.prune_method != "bi":
        model = model.to(device)

    check_model_devices(model)
    print(device)
    ppl_test = eval_ppl(args, model, tokenizer, device=device)
    logging.info(f"pre-trained model {args.model} before pruning")
    logging.info(f"wikitext perplexity {ppl_test}")

    if args.sparsity_ratio != 0:
        logging.info("pruning starts")
        if args.prune_method == "flatllm":
            prune_flatllm(args, model, tokenizer, device)
        elif args.prune_method == "bi":
            compute_bi(args, model, tokenizer, device)

    ################################################################
    if args.prune_method != 'bi':
        logging.info("*"*30)
        sparsity_ratio = check_structual_pruning(model)
        logging.info(f"sparsity sanity check {sparsity_ratio:.4f}")
    ################################################################
    
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

    print(f"num_hidden_layers: {model.config.num_hidden_layers}")
    check_model_devices(model)
    print(device)
    ppl_test = eval_ppl(args, model, tokenizer, device=device)
    logging.info(f"eigenvalue threshold {args.tol}, sparsity ratio {args.sparsity_ratio}")
    logging.info(f"pre-trained model {args.model}, ratio {args.ratio}")
    logging.info(f"wikitext perplexity {ppl_test}")
    logging.info(model)
    logging.info("*"*30)

    if args.save_model:
        # model.save_pretrained(args.save_model) # state_dict
        # torch.save(model, args.save_model)
        torch.save(model.state_dict(), args.save_model+'/pytorch_model.bin')
        tokenizer.save_pretrained(args.save_model)

    # logging.info("method\tactual_sparsity\tppl_test", file=f, flush=True)
    # logging.info(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

if __name__ == '__main__':
    main()