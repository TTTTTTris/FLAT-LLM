import time 
import heapq 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .layerwrapper import WrappedGPT, ShortLlamaHF, WrappedShortGPT
from .data import get_loaders
from .data_utils import get_dataset, prepare_dataloader, prepare_test_dataloader
import logging
import os

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention, LlamaMLP, LlamaRMSNorm
from lib.svd_llm import SVDLinear3D
import pynvml

def find_free_gpu(threshold_gb=32):
    pynvml.nvmlInit()
    num_gpus = pynvml.nvmlDeviceGetCount()

    for i in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_gb = meminfo.free / 1024 ** 3
        if free_gb >= threshold_gb:
            pynvml.nvmlShutdown()
            return i
    pynvml.nvmlShutdown()
    raise RuntimeError("No GPU with sufficient free memory found.")

# --- Your code starts here ---
def safe_inverse_on_free_gpu(cov: torch.Tensor):
    d_int = cov.shape[0]
    device_id = find_free_gpu()
    target_device = f'cuda:{device_id}'
    print(f"Using {target_device} for inversion.")

    cov = cov.to(dtype=torch.double, device=target_device)
    I = torch.eye(d_int, device=target_device, dtype=torch.double)
    ridge_inv = torch.linalg.inv(cov + I)

    return ridge_inv, cov

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def check_structual_pruning(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer, layers=[nn.Linear])

        sub_count = 0
        sub_params = 0

        for name in subset:
            out_dim = subset[name].out_features
            in_dim = subset[name].in_features
            # print(out_dim, in_dim)
            if subset[name].__class__ is nn.Linear:
                # W = subset[name].weight.data.numel()
                W = (subset[name].weight.data != 0).sum().item()               
            count += W
            sub_count += W
            total_params += in_dim * out_dim
            sub_params += in_dim * out_dim

            logging.info(f"{name} sparsity {float(W)/(in_dim * out_dim):.6f}, {W}, {in_dim * out_dim}")

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}, {float(sub_count)}, {sub_params}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, n_samples, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    print(model)

    # if "model.embed_tokens" in model.hf_device_map:
        # device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((n_samples, model.seqlen, model.config.hidden_size), dtype=dtype, device='cpu')
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            if inp.shape[0] < model.seqlen:
                pad_size = model.seqlen - inp.shape[0]
                inp = F.pad(inp, (0, 0, 0, pad_size))
            print(inp.shape, inps[0].shape)
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            data = batch[0].to(device) if type(batch) is tuple else batch.to(device)
            model(data)
        except ValueError:
            pass 
    layers[0] = layers[0].module
    # print('inps: ', inps)
    outs = torch.zeros_like(inps, device='cpu')
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def compute_bi(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad(): # inps = data; outs = 0
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, args.nsamples, dataloader, device)

    layers = model.model.layers.to('cpu')
    importances = []
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer, layers=[LlamaDecoderLayer])

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

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedShortGPT(subset[name], args=args, config=model.config, device=device)
        
        # Register hooks for attention and MLP using pre-norm inputs
        def add_batch(name, layer_id):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out[0].data, layer_id, angular=True)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name, i)))

        for j in range(args.nsamples):
            with torch.no_grad(): # input and output of current layer
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in wrapped_layers:
            importances.append(wrapped_layers[name].importances[i])
            logging.info(f"layer {i}: {wrapped_layers[name].importances[i]}")

        # update inps, outs
        for j in range(args.nsamples):
            with torch.no_grad(): # output of current layer becomes input of next layer
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer = layer.to('cpu')
        model.model.norm = model.model.norm.to('cpu')
        model.lm_head = model.lm_head.to('cpu')
        torch.cuda.empty_cache()
    
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    print(importances)
    import os
    os.makedirs(args.bi_score, exist_ok=True)
    torch.save(importances, args.bi_score + 'bi_score.pt')
    # model = ShortLlamaHF(model, args=args, config=model.config, importances=importances, n_prune_layers=args.sparsity_ratio /100 * len(layers), device=device)
    # model.remove_layers(angular=True)

def save_activations_all_layers(args, model, tokenizer, device=torch.device("cuda:0"), save_to_disk=True):
    """
    Save activations for all layers before compression.
    Args:
        save_to_disk: If True, save activations to disk and return paths; if False, return in-memory dict
    Returns a dictionary containing activations for each layer or paths to saved files.
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dtype = next(iter(model.parameters())).dtype

    print("loading calibration data for activation saving")
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
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, args.nsamples, dataloader, device)

    layers = model.model.layers.to('cpu')
    
    # Dictionary to store activations for each layer
    saved_activations = {}
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
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

        # Initialize wrapped layers to compute statistics
        wrapped_layers = {}
        for name in subset:
            if name in ['mlp.down_proj', 'self_attn.v_proj', 'self_attn.q_proj', 'self_attn.k_proj']:
                wrapped_layers[name] = WrappedGPT(subset[name], args=args, config=model.config, device=device)

        # Hooks to collect activations and compute statistics
        def add_batch(name, flag):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data, flag)
            return tmp

        handles = []
        # Register hooks for the layers we need statistics from
        for name in wrapped_layers:
            if name in ['mlp.down_proj']:
                handles.append(subset[name].register_forward_hook(add_batch(name, 0)))
            elif name in ['self_attn.v_proj', 'self_attn.q_proj', 'self_attn.k_proj']:
                handles.append(subset[name].register_forward_hook(add_batch(name, 1)))

        # Run forward passes to collect activations and compute statistics
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        
        # Remove hooks
        for h in handles:
            h.remove()

        # Initialize storage for this layer's computed statistics
        saved_activations[i] = {
            'mlp.down_proj': {'cov': None},
            'self_attn.v_proj': {'eig_vec': None}
        }
        
        # Save the computed statistics
        for name in wrapped_layers:
            if name == 'mlp.down_proj':
                saved_activations[i][name]['cov'] = wrapped_layers[name].cov.clone().cpu()
            elif name == ['self_attn.v_proj', 'self_attn.q_proj', 'self_attn.k_proj']:
                if hasattr(wrapped_layers[name], 'eig_vec'):
                    saved_activations[i][name]['eig_vec'] = wrapped_layers[name].eig_vec.clone().cpu()
        
        del wrapped_layers

        # Save to disk if requested to save memory
        if save_to_disk:
            os.makedirs(f"out_models/saved_activations/layer_{i}", exist_ok=True)
            for module_name in saved_activations[i]:
                if module_name == 'mlp.down_proj' and saved_activations[i][module_name]['cov'] is not None:
                    torch.save(saved_activations[i][module_name]['cov'], 
                              f"out_models/saved_activations/layer_{i}/{module_name}_cov.pt")
                elif module_name == 'self_attn.v_proj' and saved_activations[i][module_name]['eig_vec'] is not None:
                    torch.save(saved_activations[i][module_name]['eig_vec'], 
                              f"out_models/saved_activations/layer_{i}/{module_name}_eig_vec.pt")
            # Keep only file paths to save memory
            saved_activations[i] = {
                'mlp.down_proj': {
                    'cov': f"out_models/saved_activations/layer_{i}/mlp.down_proj_cov.pt"
                },
                'self_attn.v_proj': {
                    'eig_vec': f"out_models/saved_activations/layer_{i}/self_attn.v_proj_eig_vec.pt"
                }
            }

        # Update inps for next layer
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer = layer.to('cpu')
        torch.cuda.empty_cache()
        
        print(f"Saved activations for layer {i}")

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    
    return saved_activations

@torch.no_grad()
def prune_flatllm_with_precomputed_stats(args, model, tokenizer, device=torch.device("cuda:0"), save_to_disk=True):
    """
    Alternative name for the new compression approach using pre-computed statistics.
    """
    return prune_flatllm_with_saved_activations(args, model, tokenizer, device, save_to_disk)

@torch.no_grad()  
def prune_flatllm_with_saved_activations(args, model, tokenizer, device=torch.device("cuda:0"), save_to_disk=True):
    """
    Main compression function that first saves all activations, then performs compression.
    This is the new recommended approach.
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dtype = next(iter(model.parameters())).dtype

    # First, save activations for all layers
    print("Saving activations for all layers before compression...")
    # saved_activations = save_activations_all_layers(args, model, tokenizer, device, save_to_disk)
    print("Activation saving complete")

    print("loading calibdation data")
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
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, args.nsamples, dataloader, device)

    layers = model.model.layers.to('cpu')

    if args.bi_score != None:
        bi_score = torch.load(args.bi_score + 'sparsity_score_' + str(args.sparsity_ratio) + '%.pt', weights_only=False)
    else:
        bi_score = torch.ones(len(layers)).to(device) * args.sparsity_ratio / 100
    print(bi_score)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
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

        wrapped_layers = {}
        if bi_score[i] < 1:
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name], args=args, config=model.config, device=device)

        # Use saved pre-computed statistics instead of forward hooks
        if bi_score[i] < 1:
            for name in wrapped_layers:
                if name in ['mlp.down_proj']:
                    # Load pre-computed covariance matrix
                    saved_cov = torch.load(f'out_models/saved_activations/layer_{i}/{name}_cov.pt')
                    wrapped_layers[name].load_precomputed_statistics(saved_cov=saved_cov)
                elif name in ['self_attn.v_proj']:
                    # Load pre-computed eigenvectors
                    saved_eig_vec = torch.load(f'out_models/saved_activations/layer_{i}/{name}_eig_vec.pt')
                    wrapped_layers[name].load_precomputed_statistics(saved_eig_vec=saved_eig_vec)

        # No forward pass needed here when using precomputed statistics
        # The forward pass will be done at the end to get outputs for the next layer
        torch.cuda.empty_cache()

        # Rest of the compression logic remains the same...
        if bi_score[i] < 1:
            for name in subset:
                if name in ['mlp.down_proj']:
                    shape = subset[name].weight.shape
                    eig_num = int(bi_score[i] * shape[1])
                    logging.info(f"structural pruning layer {i}, {name}:, {eig_num}, {shape[1]}")
                    W2 = subset['mlp.down_proj'].weight.to(torch.double)
                    W1 = subset['mlp.up_proj'].weight
                    W0 = subset['mlp.gate_proj'].weight

                    if torch.cuda.device_count() > 1:
                        ridge_inv, cov = safe_inverse_on_free_gpu(wrapped_layers[name].cov)
                    else:
                        cov = wrapped_layers[name].cov.to(torch.double)
                        d_int = cov.shape[0]
                        ridge_inv = torch.linalg.inv(cov + torch.eye(d_int, device=cov.device))

                    scores = torch.diagonal(cov @ ridge_inv)
                    idx = torch.topk(scores, k=eig_num, largest=True).indices
                    del ridge_inv, scores
                    torch.cuda.empty_cache()

                    idx = idx.to(device=cov.device)
                    middle = torch.linalg.inv(cov[idx][:, idx])

                    if torch.cuda.device_count() > 1:
                        device_id = find_free_gpu()
                        target_device = f'cuda:{device_id}'
                        print(f"Using {target_device} for multiplication.")
                        middle = middle.to(target_device)
                        cov = cov.to(target_device)
                        idx = idx.to(target_device)
                        W2 = W2.to(target_device)

                    W2 = (middle @ (cov[idx])  @ (W2.T)).T
                    
                    idx = idx.to(W0.device)
                    W1 = W1[idx, :]   
                    W0 = W0[idx, :] 

                    subset['mlp.down_proj'].__dict__.pop('weight', None)
                    subset['mlp.up_proj'].__dict__.pop('weight', None)
                    subset['mlp.gate_proj'].__dict__.pop('weight', None)

                    subset['mlp.down_proj'].weight = torch.nn.Parameter(W2.to(W0.device, torch.float16))
                    subset['mlp.up_proj'].weight = torch.nn.Parameter(W1.to(torch.float16))
                    subset['mlp.gate_proj'].weight = torch.nn.Parameter(W0.to(torch.float16))

                    del W0, W1, W2, middle, cov, idx
                torch.cuda.empty_cache()

                if name in ['self_attn.v_proj']:
                    print('Nheads:', model.config.num_key_value_heads, model.config.num_attention_heads)
                    shape = subset[name].weight.shape
                    if model.config.num_key_value_heads == model.config.num_attention_heads:
                        N_head = model.config.num_attention_heads
                        d_head = shape[0] // N_head
                        eig_num = torch.tensor([int(bi_score[i] * d_head)] * N_head)
                        logging.info(f"structural pruning layer {i}, {name}:, {eig_num}, {shape[0]}")
                        Q = wrapped_layers[name].eig_vec.to(torch.double).to(device)
                        Qr = torch.stack([
                            Q[i, :, :eig_num[i]]
                            for i in range(Q.shape[0])
                        ], dim=0)

                        Wv = subset[name].weight.reshape(
                            N_head, d_head, -1).to(torch.double)
                        Wo = subset['self_attn.o_proj'].weight.reshape(
                            -1, N_head, d_head).transpose(0, 1).to(torch.double)
                        Wv = torch.bmm(Qr.transpose(1, 2), Wv).reshape(
                            -1, shape[-1])
                        Wo = torch.bmm(Wo, Qr).transpose(0, 1).reshape(
                            shape[-1], -1)

                        subset['self_attn.v_proj'].weight = torch.nn.Parameter(Wv.to(torch.float16))
                        subset['self_attn.o_proj'].weight = torch.nn.Parameter(Wo.to(torch.float16))

                        del Q, Qr, Wv, Wo
                    else:
                        # Handle GQA case...
                        N_head_kv = model.config.num_key_value_heads
                        N_head = model.config.num_attention_heads
                        d_head = shape[0] // N_head_kv
                        group_size = N_head // N_head_kv 

                        eig_num = torch.tensor([int(bi_score[i] * d_head)] * N_head_kv)
                        logging.info(f"structural pruning layer {i}, {name}:, {eig_num}, {shape[0]}")
                        Q = wrapped_layers[name].eig_vec.to(torch.double).to(device)
                        Qr = torch.stack([
                            Q[i, :, :eig_num[i]]
                            for i in range(Q.shape[0])
                        ], dim=0)

                        Wv = subset[name].weight.reshape(
                            N_head_kv, d_head, -1).to(torch.double)
                        Wo = subset['self_attn.o_proj'].weight.reshape(
                            -1, N_head, d_head).transpose(0, 1).to(torch.double)
                        Qr = Qr.to(Wv.device)
                        Wv = torch.bmm(Qr.transpose(1, 2), Wv).reshape(
                            -1, shape[-1])

                        kv_indices = torch.arange(N_head, device=device) // group_size
                        kv_indices = kv_indices.to('cpu')

                        Qr_o = Qr[kv_indices]
                        Qr_o = Qr_o.to(Wo.device)

                        Wo = torch.bmm(Wo, Qr_o).transpose(0, 1).reshape(
                            shape[-1], -1)
                        
                        subset['self_attn.v_proj'].__dict__.pop('weight', None)
                        subset['self_attn.o_proj'].__dict__.pop('weight', None)

                        subset['self_attn.v_proj'].weight = torch.nn.Parameter(Wv.to(dtype))
                        subset['self_attn.o_proj'].weight = torch.nn.Parameter(Wo.to(dtype))

                        del Q, Qr, Qr_o, Wv, Wo, kv_indices
                torch.cuda.empty_cache()

        del wrapped_layers
        torch.cuda.empty_cache()
        
        # No need to update inputs for next layer when using precomputed statistics
        # and not requiring sequential processing
        layer = layer.to('cpu')
        print(f"Compressed layer {i}")

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
@torch.no_grad()
def prune_flatllm(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dtype = next(iter(model.parameters())).dtype

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

    layers = model.model.layers.to('cpu')

    if args.bi_score != None:
        bi_score = torch.load(args.bi_score + 'sparsity_score_' + str(args.sparsity_ratio) + '%.pt', weights_only=False)
    else:
        bi_score = torch.ones(len(layers)).to(device) * args.sparsity_ratio / 100
    print(bi_score)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

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

        if inps is None:
            print('Warning: inps is None')
        if outs is None:
            print('Warning: outs is None')
        if attention_mask is None:
            print('Warning: attention_mask is None')
        if position_ids is None:
            print('Warning: position_ids is None')

        wrapped_layers = {}
        if bi_score[i] < 1:
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name], args=args, config=model.config, device=device)

        def add_batch(name, flag):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data, flag)
            return tmp

        handles = []

        if bi_score[i] < 1:
            for name in wrapped_layers:
                if name in ['mlp.down_proj']:
                    handles.append(subset[name].register_forward_hook(add_batch(name, 0)))
                elif name in ['self_attn.v_proj', 'self_attn.q_proj', 'self_attn.k_proj']:
                    handles.append(subset[name].register_forward_hook(add_batch(name, 1)))

        for j in range(args.nsamples):
            with torch.no_grad(): # input and output of current layer
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()

        if bi_score[i] < 1:
            for name in subset:
                shape = subset[name].weight.shape
                if name in ['mlp.down_proj']:
                    # size = math.prod(shape)
                    # eig_num = wrapped_layers[name].eig_num
                    eig_num = int(bi_score[i] * shape[1])
                    # reduced_size = eig_num * (shape[0] + shape[1])
                    # if(reduced_size < size):
                    logging.info(f"structural pruning layer {i}, {name}:, {eig_num}, {shape[1]}")
                    W2 = subset['mlp.down_proj'].weight.to(torch.double)
                    W1 = subset['mlp.up_proj'].weight
                    W0 = subset['mlp.gate_proj'].weight

                    if torch.cuda.device_count() > 1:
                        ridge_inv, cov = safe_inverse_on_free_gpu(wrapped_layers[name].cov)
                    else:
                        cov = wrapped_layers[name].cov.to(torch.double)
                        d_int = cov.shape[0]
                        ridge_inv = torch.linalg.inv(cov + torch.eye(d_int, device=cov.device))

                    # Step 3: Compute ridge leverage scores
                    scores = torch.diagonal(cov @ ridge_inv)  # [d_int]

                    # Step 4: Select top-k neuron indices
                    idx = torch.topk(scores, k=eig_num, largest=True).indices

                    del ridge_inv, scores
                    torch.cuda.empty_cache()

                    # Replace full Sk with indexed matmul
                    idx = idx.to(device=cov.device)

                    middle = torch.linalg.inv(cov[idx][:, idx])

                    # Project weights in reduced basis
                    if torch.cuda.device_count() > 1:
                        device_id = find_free_gpu()
                        target_device = f'cuda:{device_id}'
                        print(f"Using {target_device} for multiplication.")
                        middle = middle.to(target_device)
                        cov = cov.to(target_device)
                        idx = idx.to(target_device)
                        W2 = W2.to(target_device)

                    W2 = (middle @ (cov[idx])  @ (W2.T)).T  # [d_h, k]
                    
                    idx = idx.to(W0.device)
                    W1 = W1[idx, :]   
                    W0 = W0[idx, :] 

                    subset['mlp.down_proj'].__dict__.pop('weight', None)
                    subset['mlp.up_proj'].__dict__.pop('weight', None)
                    subset['mlp.gate_proj'].__dict__.pop('weight', None)

                    # identity
                    # print(Sk @ torch.linalg.pinv(Sk.T @ wrapped_layers[name].cov @ Sk) @ Sk.T @ wrapped_layers[name].cov)

                    subset['mlp.down_proj'].weight = torch.nn.Parameter(W2.to(W0.device, torch.float16))
                    subset['mlp.up_proj'].weight = torch.nn.Parameter(W1.to(torch.float16))
                    subset['mlp.gate_proj'].weight = torch.nn.Parameter(W0.to(torch.float16))

                    del W0, W1, W2, middle, cov, idx
                torch.cuda.empty_cache()

                if name in ['self_attn.v_proj']: # flatllm
                    if model.config.num_key_value_heads == model.config.num_attention_heads:
                        N_head = model.config.num_attention_heads
                        d_head = shape[0] // N_head
                        eig_num = torch.tensor([int(bi_score[i] * d_head)] * N_head)
                        # max_eig_num = eig_num.max().item()  # Find the maximum eig_num to slice up to the largest value
                        logging.info(f"structural pruning layer {i}, {name}:, {eig_num}, {shape[0]}")
                        Q = wrapped_layers[name].eig_vec.to(torch.double).to(device) # [head, d_h, d_h]
                        Qr = torch.stack([
                            Q[i, :, :eig_num[i]]
                            for i in range(Q.shape[0])
                        ], dim=0) # [head, d_h, r] [32, 128, 18]

                        # Per-head orthogonal Q
                        Wv = subset[name].weight.reshape(
                            N_head, d_head, -1).to(torch.double) # [head, d_h, d]
                        Wo = subset['self_attn.o_proj'].weight.reshape(
                            -1, N_head, d_head).transpose(0, 1).to(torch.double) # [d, head, d_h]
                        Wv = torch.bmm(Qr.transpose(1, 2), Wv).reshape(
                            -1, shape[-1])  # [32, 18, 128] x [32, 128, 4096] -> [32, 18, 4096]
                        Wo = torch.bmm(Wo, Qr).transpose(0, 1).reshape(
                            shape[-1], -1)

                        subset['self_attn.v_proj'].weight = torch.nn.Parameter(Wv.to(torch.float16))
                        # subset['self_attn.v_proj'].out_features = Wv.shape[0]
                        subset['self_attn.o_proj'].weight = torch.nn.Parameter(Wo.to(torch.float16))
                        # subset['self_attn.o_proj'].in_features = Wo.shape[1]

                        del Q, Qr, Wv, Wo
                    else:
                        N_head_kv = model.config.num_key_value_heads
                        N_head = model.config.num_attention_heads
                        d_head = shape[0] // N_head_kv
                        group_size = N_head // N_head_kv 

                        eig_num = torch.tensor([int(bi_score[i] * d_head)] * N_head_kv)
                        # max_eig_num = eig_num.max().item()  # Find the maximum eig_num to slice up to the largest value
                        logging.info(f"structural pruning layer {i}, {name}:, {eig_num}, {shape[0]}")
                        Q = wrapped_layers[name].eig_vec.to(torch.double).to(device) # [head, d_h, d_h]
                        Qr = torch.stack([
                            Q[i, :, :eig_num[i]]
                            for i in range(Q.shape[0])
                        ], dim=0) # [head, d_h, r] [32, 128, 18]
                        # Qr = torch.stack([
                        #         F.pad(Q[i, :, :eig_num[i].item()], (0, max_eig_num - eig_num[i].item()))
                        #         for i in range(Q.shape[0])
                        #     ], dim=0)

                        # Per-head orthogonal Q
                        Wv = subset[name].weight.reshape(
                            N_head_kv, d_head, -1).to(torch.double) # [head, d_h, d]
                        Wo = subset['self_attn.o_proj'].weight.reshape(
                            -1, N_head, d_head).transpose(0, 1).to(torch.double) # [d, head, d_h]
                        Qr = Qr.to(Wv.device)
                        Wv = torch.bmm(Qr.transpose(1, 2), Wv).reshape(
                            -1, shape[-1])  # [32, 18, 128] x [32, 128, 4096] -> [32, 18, 4096]

                        kv_indices = torch.arange(N_head, device=device) // group_size
                        kv_indices = kv_indices.to('cpu')

                        Qr_o = Qr[kv_indices]
                        Qr_o = Qr_o.to(Wo.device)

                        Wo = torch.bmm(Wo, Qr_o).transpose(0, 1).reshape(
                            shape[-1], -1)
                        
                        subset['self_attn.v_proj'].__dict__.pop('weight', None)
                        subset['self_attn.o_proj'].__dict__.pop('weight', None)

                        subset['self_attn.v_proj'].weight = torch.nn.Parameter(Wv.to(dtype))
                        # subset['self_attn.v_proj'].out_features = Wv.shape[0]
                        subset['self_attn.o_proj'].weight = torch.nn.Parameter(Wo.to(dtype))
                        # subset['self_attn.o_proj'].in_features = Wo.shape[1]

                        del Q, Qr, Qr_o, Wv, Wo, kv_indices
                torch.cuda.empty_cache()  # Explicitly clear GPU memory

                if name in ['self_attn.q_proj', 'self_attn.k_proj']:
                    if shape[0] == shape[1]:
                        N_head = model.config.num_attention_heads
                        d_head = shape[0] // N_head
                        eig_num = torch.tensor([int(bi_score[i] * d_head)] * N_head)
                        # max_eig_num = eig_num.max().item()  # Find the maximum eig_num to slice up to the largest value
                        logging.info(f"structural pruning layer {i}, {name}:, {eig_num}, {shape[0]}")
                        factor = subset[name].weight.to(torch.double)
                        Q = wrapped_layers[name].eig_vec.to(torch.double).to(device) # [head, d_h, d_h]
                        Qr = torch.stack([
                                Q[i, :, :eig_num[i]]
                                for i in range(Q.shape[0])
                            ], dim=0) # [head, d_h, r] [32, 128, 18]
                        # Create an SVDLinear layer
                        svd_layer = SVDLinear3D(num_heads=N_head, d_head=int(shape[0] / N_head), 
                                                rank=eig_num[0], device=device, dtype=torch.float16)
                        # print(svd_layer.u_proj.data.shape, svd_layer.v_proj.data.shape, Qr.shape, factor.shape, N_head, d_head) 
                        # torch.Size([32, 128, 126]) torch.Size([4032, 4096]) torch.Size([8, 512, 126]) torch.Size([4096, 4096]) 32 128
                        svd_layer.u_proj.data = Qr.to(torch.float16) # [32, 128, 18]
                        svd_layer.v_proj.data = torch.bmm(Qr.transpose(1,2), factor.reshape( # [32, 18, 128] x [32, 128, 4096] -> [32, 18, 4096]
                                                    N_head, d_head, -1)).to(torch.float16).reshape(-1, shape[-1])
                        # Replace the TLinear layer with the new nn.Linear layer
                        parent_module, child_name = layer, name
                        if '.' in name:
                            *parent_path, child_name = name.split('.')
                            for part in parent_path:
                                parent_module = getattr(parent_module, part)
                        setattr(parent_module, child_name, svd_layer)

                        del factor, Q, Qr, svd_layer
                    else:
                        N_head_kv = model.config.num_key_value_heads
                        d_head = shape[0] // N_head_kv

                        eig_num = torch.tensor([int(bi_score[i] * d_head)] * N_head_kv)
                        # max_eig_num = eig_num.max().item()  # Find the maximum eig_num to slice up to the largest value
                        logging.info(f"structural pruning layer {i}, {name}:, {eig_num}, {shape[0]}")
                        factor = subset[name].weight.to(torch.double)
                        Q = wrapped_layers[name].eig_vec.to(torch.double).to(device) # [head, d_h, d_h]
                        Qr = torch.stack([
                                Q[i, :, :eig_num[i]]
                                for i in range(Q.shape[0])
                            ], dim=0) # [head, d_h, r] [32, 128, 18]
                        # Create an SVDLinear layer
                        svd_layer = SVDLinear3D(num_heads=N_head_kv, d_head=int(shape[0] / N_head_kv), 
                                                rank=eig_num[0], device=device, dtype=torch.float16)
                        svd_layer.u_proj.data = Qr.to(torch.float16) # [32, 128, 18]
                        svd_layer.v_proj.data = torch.bmm(Qr.transpose(1,2), factor.reshape( # [32, 18, 128] x [32, 128, 4096] -> [32, 18, 4096]
                                                    N_head_kv, d_head, -1)).to(torch.float16).reshape(-1, shape[-1])
                        # Replace the TLinear layer with the new nn.Linear layer
                        parent_module, child_name = layer, name
                        if '.' in name:
                            *parent_path, child_name = name.split('.')
                            for part in parent_path:
                                parent_module = getattr(parent_module, part)
                        setattr(parent_module, child_name, svd_layer)

                        del factor, Q, Qr, svd_layer
                torch.cuda.empty_cache()  # Explicitly clear GPU memory

                # if name in ['self_attn.v_proj']: # modegpt
                #     print('Nheads:', model.config.num_key_value_heads, model.config.num_attention_heads)
                #     shape = subset[name].weight.shape
                #     if model.config.num_key_value_heads == model.config.num_attention_heads:
                #     # eig_num = wrapped_layers[name].eig_num
                #         N_head = model.config.num_attention_heads
                #         d_head = shape[0] // N_head
                #         eig_num = torch.tensor(int(bi_score[i] * d_head))
                #         # max_eig_num = eig_num.max().item()  # Find the maximum eig_num to slice up to the largest value
                #         logging.info(f"structural pruning layer {i}, {name}:, {eig_num}, {d_head}")
                        
                #         cov = wrapped_layers[name].cov.to(torch.double).to(device) # [head, d_h, d_h]
                #         eig_val, eig_vec = torch.linalg.eigh(cov)
                #         eig_val_sqrt = torch.sqrt(torch.clamp(eig_val,0))
                #         cov_sqrt = eig_vec @ torch.diag_embed(eig_val_sqrt) @ eig_vec.transpose(-2, -1)
                #         eig_val_sqrt_inv = 1 / (eig_val_sqrt + 1e-8)
                #         cov_sqrt_inv = eig_vec @ torch.diag_embed(eig_val_sqrt_inv) @ eig_vec.transpose(-2, -1)

                #         Wv = subset['self_attn.v_proj'].weight.transpose(0,1).reshape(
                #             -1, N_head, d_head).transpose(0, 1).to(torch.double) # [head, d_h, d]
                #         Wo = subset['self_attn.o_proj'].weight.transpose(0,1).reshape(
                #             N_head, d_head, -1).to(torch.double) # [head, d, d_h] 
                #         U, S, V = torch.linalg.svd(cov_sqrt @ Wv, full_matrices=False) # [head, d_h, d]
                #         U1, S1, V1 = torch.linalg.svd(torch.diag_embed(S) @ V @ Wo, full_matrices=False) # [head, d, d_h]

                #         # Per-head orthogonal Q
                #         Wv = cov_sqrt_inv @ U @ U1[:,:,:eig_num]
                #         Wo = torch.diag_embed(S1)[:,:eig_num,:eig_num] @  V1[:,:eig_num,:]
                #         print(eig_num, Wv.shape, Wo.shape)
                #         Wv = Wv.transpose(0,1).reshape(
                #             shape[-1], -1).transpose(0,1) 
                #         Wo = Wo.reshape(
                #             -1, shape[-1]).transpose(0,1)
                        
                #         subset['self_attn.v_proj'].__dict__.pop('weight', None)
                #         subset['self_attn.o_proj'].__dict__.pop('weight', None)
                #         subset['self_attn.v_proj'].weight = torch.nn.Parameter(Wv.to(torch.float16))
                #         # subset['self_attn.v_proj'].out_features = Wv.shape[0]
                #         subset['self_attn.o_proj'].weight = torch.nn.Parameter(Wo.to(torch.float16))
                #         # subset['self_attn.o_proj'].in_features = Wo.shape[1]

                #         del cov, eig_val, eig_vec, cov_sqrt, cov_sqrt_inv, U,S,V,U1,S1,V1, Wv, Wo
                # torch.cuda.empty_cache()  # Explicitly clear GPU memory

                # if name in ['self_attn.q_proj']:
                #     if model.config.num_key_value_heads == model.config.num_attention_heads:
                #         shape = subset[name].weight.shape
                #         N_head_kv = model.config.num_key_value_heads
                #         N_head = model.config.num_attention_heads
                #         d_head = shape[0] // N_head
                #         group_size = N_head // N_head_kv 

                #         eig_num = torch.tensor(int(bi_score[i] * d_head))
                #         # max_eig_num = eig_num.max().item()  # Find the maximum eig_num to slice up to the largest value
                #         logging.info(f"structural pruning layer {i}, {name}:, {eig_num}, {shape[0]}")

                #         scores = torch.norm(wrapped_layers['self_attn.rope_module'].cov_sqrt_q, dim=1) * torch.norm(wrapped_layers['self_attn.rope_module'].cov_sqrt_k, dim=-1)
                #         idx = torch.topk(scores, k=eig_num, largest=True, dim=-1).indices
                #         print(scores)
                #         print(idx)
                #         idx_expanded = idx.unsqueeze(-1).expand(-1, -1, shape[1])  # [N_head, k, d]
                        
                #         # Store pruning indices in the RoPE module for later use
                #         if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rope_module'):
                #             # For regular MHA, no kv_indices mapping needed
                #             layer.self_attn.rope_module.set_pruning_indices(idx, None)
                        
                #         # Per-head orthogonal Q
                #         Wq = subset['self_attn.q_proj'].weight.reshape(
                #             N_head, d_head, -1) # [head, d_h, d]
                #         Wk = subset['self_attn.k_proj'].weight.reshape(
                #             N_head_kv, d_head, -1) # [head, d_h, d]
                #         # for h in range(idx.shape[0]):
                #         #     Wq[h] = Wq[h, idx[h], :]
                #         #     Wk[h] = Wk[h, idx[h], :]
                #         Wq = torch.gather(Wq, 1, idx_expanded)  # [N_head, k, d]
                #         Wk = torch.gather(Wk, 1, idx_expanded)  # [N_head, k, d]

                #         print(Wq.shape, Wk.shape)

                #         subset['self_attn.q_proj'].__dict__.pop('weight', None)
                #         subset['self_attn.k_proj'].__dict__.pop('weight', None)
                #         subset['self_attn.q_proj'].weight = torch.nn.Parameter(Wq.reshape(
                #             -1, shape[1]).to(dtype))
                #         subset['self_attn.k_proj'].weight = torch.nn.Parameter(Wk.reshape(
                #             -1, shape[1]).to(dtype))

                #         del Wq, Wk, scores, idx, idx_expanded
                    
                #     else:
                #         shape = subset[name].weight.shape
                #         N_head_kv = model.config.num_key_value_heads
                #         N_head = model.config.num_attention_heads
                #         d_head = shape[0] // N_head
                #         group_size = N_head // N_head_kv 

                #         eig_num = torch.tensor(int(bi_score[i] * d_head))
                #         # max_eig_num = eig_num.max().item()  # Find the maximum eig_num to slice up to the largest value
                #         logging.info(f"structural pruning layer {i}, {name}:, {eig_num}, {shape[0]}")

                #         # Store pruning indices in the RoPE module for later use  
                #         kv_indices = torch.arange(N_head, device='cpu') // group_size
                            
                #         scores = torch.norm(wrapped_layers['self_attn.rope_module'].cov_sqrt_q, dim=1) * torch.norm(wrapped_layers['self_attn.rope_module'].cov_sqrt_k, dim=-1)[kv_indices]
                #         scores = scores.reshape(group_size, N_head_kv, -1).sum(dim=0)
                #         idx = torch.topk(scores, k=eig_num, largest=True, dim=-1).indices
                #         print(scores)
                #         print(idx)

                #         layer.self_attn.rope_module.set_pruning_indices(idx, kv_indices)

                #         Wq = subset['self_attn.q_proj'].weight.reshape(
                #             N_head, d_head, -1) # [head, d_h, d]
                #         Wk = subset['self_attn.k_proj'].weight.reshape(
                #             N_head_kv, d_head, -1) # [head, d_h, d]
                #         # for h in range(idx.shape[0]):
                #         #     Wq[h] = Wq[h, idx[h], :]
                #         #     Wk[h] = Wk[h, idx[h], :]
                #         Wq = torch.gather(Wq, 1, idx[kv_indices].unsqueeze(-1).expand(-1, -1, shape[1]))  # [N_head, k, d]
                #         Wk = torch.gather(Wk, 1, idx.unsqueeze(-1).expand(-1, -1, shape[1]))  # [N_head, k, d]

                #         print(Wq.shape, Wk.shape)

                #         subset['self_attn.q_proj'].__dict__.pop('weight', None)
                #         subset['self_attn.k_proj'].__dict__.pop('weight', None)
                #         subset['self_attn.q_proj'].weight = torch.nn.Parameter(Wq.reshape(
                #             -1, shape[1]).to(dtype))
                #         subset['self_attn.k_proj'].weight = torch.nn.Parameter(Wk.reshape(
                #             -1, shape[1]).to(dtype))

                #         del Wq, Wk, scores, idx
                #     torch.cuda.empty_cache()  # Explicitly clear GPU memory

        del wrapped_layers
        torch.cuda.empty_cache()  # Explicitly clear GPU memory
        for j in range(args.nsamples):
            with torch.no_grad(): # output of current layer becomes input of next layer
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        torch.cuda.empty_cache()
        layer = layer.to('cpu')

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
