import time 
import heapq 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .layerwrapper import WrappedGPT, ShortLlamaHF, WrappedShortGPT
from .data import get_loaders
from .data_utils import get_dataset, prepare_dataloader, prepare_test_dataloader
import logging

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention, LlamaMLP, LlamaRMSNorm

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
        print(batch)
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

@torch.no_grad()
def prune_flatllm(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dtype = next(iter(model.parameters())).dtype

    if args.bi_score != None:
        bi_score = torch.load(args.bi_score + 'sparsity_score_' + str(args.sparsity_ratio) + '%.pt', weights_only=False)
    else:
        bi_score = torch.ones(len(layers)).to(device) * args.sparsity_ratio / 100
    print(bi_score)

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
                elif name in ['self_attn.v_proj']:
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

                if name in ['self_attn.v_proj']:
                    print('Nheads:', model.config.num_key_value_heads, model.config.num_attention_heads)
                    if model.config.num_key_value_heads == model.config.num_attention_heads:
                    # eig_num = wrapped_layers[name].eig_num
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
