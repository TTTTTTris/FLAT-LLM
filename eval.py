import torch
import itertools
import time
import os
import argparse
import logging
import torch.nn as nn
from ml_collections import ConfigDict
from slicegpt import data_utils, gpu_utils, hf_utils, utils


config = ConfigDict()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def benchmarking_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    path_group.add_argument(
        "--sliced-model-path",
        type=str,
        help="Path to load the model to fine-tune (sliced) and tokenizer from",
        default=None,
    )

    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument(
        "--eval-dataset",
        type=str,
        help="Dataset to evaluate on.",
        choices=["wikitext2", "ptb", "c4"],
        default="wikitext2",
    )
    parser.add_argument(
        "--ntokens",
        type=int,
        help="Number of tokens to benchmark over.",
        default=128,
    )
    parser.add_argument("--seqlen", type=int, default=256, help="Sequence length for loading the calibration data.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for loading the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))

    parser.add_argument('--wandb-project', type=str, default="slicegpt-bench", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )

    return parser.parse_args() if interactive else parser.parse_args('')

def process_benchmarking_args(args: argparse.Namespace):
    logging.debug(f'Parsed arguments:')
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    # if not 0 <= args.sparsity < 1:
        # raise argparse.ArgumentTypeError(f"Sparsity should be in the range [0, 1)")

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")

from datasets import load_dataset
from torch.utils.data.dataset import Dataset
def get_test_data(name, tokenizer, seq_len=2048, batch_size = 4):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)
    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    ####
    if 'wikitext2' in name:
        test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

@torch.no_grad()
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=1, generated_len=128, batch_size=32, device="cuda"):
    model.eval()
    throughput = 0
    token_num = 0
    end_memory = 0
    # num_batches_to_fetch = 4
    print(original_len, generated_len, batch_size)
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size = batch_size)
    weight_memory = torch.cuda.memory_allocated()
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, 1)):
        batch = batch_data.to(device)
        token_num += batch.shape[0] * generated_len
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        print(start_memory, weight_memory)
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start_time = time.time()
        generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True,
                max_new_tokens=generated_len,
                # top_k=50,
                # max_length=original_len+generated_len,
                # top_p=0.95,
                # temperature=1,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = max(torch.cuda.max_memory_allocated(0), end_memory)
        if torch.isfinite(generation_output[0]).all():  # check if the generation is successful since fp16 may cause nan
            throughput += end_time - start_time
            print("time: {}".format(end_time - start_time))
    print("Total Memory: {} GB".format(end_memory/(1024 ** 3)))
    print("Weight Memory: {} GB".format(weight_memory/(1024 ** 3)))
    print("Activation Memory: {} GB".format((end_memory - start_memory)/(1024 ** 3)))
    print("Throughput: {} tokens/sec".format(token_num / throughput))

layer_times = {}

def register_timing_hooks(model):
    def make_hook(name):
        def hook(module, input, output):
            torch.cuda.synchronize()
            start = time.time()
            torch.cuda.synchronize()
            end = time.time()
            if name not in layer_times:
                layer_times[name] = []
            layer_times[name].append(end - start)
        return hook

    for name, module in model.named_modules():
        if "self_attn" in name or "mlp" in name:
            module.register_forward_hook(make_hook(name))

@torch.no_grad()
def eff_eval_with_prefill_decode(
    model, tokenizer, dataset='wikitext2',
    original_len=1024, generated_len=256,
    batch_size=4, device="cuda"
):
    register_timing_hooks(model)
    model.eval()

    prefill_times, decode_times = [], []
    activation_mem_prefill, activation_mem_decode = [], []
    weight_mem_reserved = []

    num_batches_to_fetch = 10
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size=batch_size)

    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)

        # === PREFILL ===
        torch.cuda.synchronize()
        prefill_start = time.time()

        outputs = model(input_ids=batch, use_cache=True)
        past_key_values = outputs.past_key_values

        torch.cuda.synchronize()
        prefill_end = time.time()
        prefill_time = prefill_end - prefill_start
        prefill_times.append(prefill_time)

        # Memory usage after prefill
        activation_mem_prefill.append(torch.cuda.max_memory_allocated(device) / 1e6)  # in MB
        weight_mem_reserved.append(torch.cuda.max_memory_reserved(device) / 1e6)     # in MB

        print(f"[Batch {batch_idx}] Prefill time: {prefill_time:.4f}s | Activation memory: {activation_mem_prefill[-1]:.2f}MB")

        # === DECODE ===
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        decode_start = time.time()

        current_ids = batch
        last_token = current_ids[:, -1:]

        for _ in range(generated_len):
            outputs = model(input_ids=last_token, past_key_values=past_key_values, use_cache=True)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            last_token = next_token
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

        torch.cuda.synchronize()
        decode_end = time.time()
        decode_time = decode_end - decode_start
        decode_times.append(decode_time)

        activation_mem_decode.append(torch.cuda.max_memory_allocated(device) / 1e6)  # in MB

        print(f"[Batch {batch_idx}] Decode time: {decode_time:.4f}s | Activation memory: {activation_mem_decode[-1]:.2f}MB")

    print("\n=== Layer Runtime Summary ===")
    for name, times in layer_times.items():
        if times:
            avg = sum(times) / len(times)
            print(f"{name}: {avg*1000:.4f} ms (avg over {len(times)} runs)")

    # === Aggregate Results ===
    avg_prefill_time = sum(prefill_times) / len(prefill_times)
    avg_decode_time = sum(decode_times) / len(decode_times)

    prefill_throughput = (original_len * batch_size) / avg_prefill_time
    decode_throughput = (generated_len * batch_size) / avg_decode_time

    print("\n=== FINAL RESULTS ===")
    print(f"Prefill throughput: {prefill_throughput:.2f} tokens/sec")
    print(f"Decode throughput: {decode_throughput:.2f} tokens/sec")
    print(f"Avg prefill time: {avg_prefill_time:.4f}s | Avg activation memory: {sum(activation_mem_prefill)/len(activation_mem_prefill):.2f}MB")
    print(f"Avg decode time: {avg_decode_time:.4f}s | Avg activation memory: {sum(activation_mem_decode)/len(activation_mem_decode):.2f}MB")
    print(f"Max reserved (weights) memory: {max(weight_mem_reserved):.2f}MB")

    return {
        "avg_prefill_time": avg_prefill_time,
        "avg_decode_time": avg_decode_time,
        "prefill_throughput": prefill_throughput,
        "decode_throughput": decode_throughput,
        "activation_memory_prefill_MB": activation_mem_prefill,
        "activation_memory_decode_MB": activation_mem_decode,
        "weights_reserved_memory_MB": weight_mem_reserved,
    }

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
    for key in model:
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
            new_layer = nn.Linear(in_features, out_features, bias=bias).to(dtype=weight.dtype)
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

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
if __name__ == "__main__":
    utils.configure_logging()
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    benchmarking_args = benchmarking_arg_parser()
    process_benchmarking_args(benchmarking_args)
    model = AutoModelForCausalLM.from_pretrained(
        benchmarking_args.model, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(benchmarking_args.model, use_fast=True)
    model.config._attn_implementation = "sdpa"
    # model = replace_decoder_layers(model)             
    print(f"Attention implementation: {getattr(model.config, '_attn_implementation', 'not specified')}")
    if benchmarking_args.sliced_model_path:
        state_dict= torch.load(benchmarking_args.sliced_model_path, weights_only=False, map_location="cpu")
        if isinstance(state_dict, dict):
            resize_and_substitute_linear_layers(model, state_dict)
            # Load the full state dict without shape checking
            model.load_state_dict(state_dict, strict=False)
            model.to("cuda", non_blocking=True)
            torch.cuda.empty_cache()

        else:
            del model
            torch.cuda.empty_cache()
            model = state_dict.to(config.device)
        print(f"Loaded model from {benchmarking_args.sliced_model_path}")
        try:
            tokenizer = torch.load(benchmarking_args.sliced_model_path, weights_only=False)['tokenizer']
        except Exception as e:
            print(f"Failed to load tokenizer from {benchmarking_args.sliced_model_path}: {e}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify all parameters are on the same device
    devices = {param.device for param in model.parameters()}
    dtype = {param.dtype for param in model.parameters()}
    print(f"Model devices: {devices}")
    print(f"Model dtype: {dtype}")
    
    if len(devices) > 1:
        print("WARNING: Model parameters are on multiple devices!")
        # Force all parameters to the same device
        model = model.to(config.device)
        print(f"Moved all parameters to {config.device}")
    if len(dtype) > 1:
        print("WARNING: Model parameters have multiple dtypes!")
        # Force all parameters to the same device
        model = model.to(config.dtype)
        print(f"Moved all parameters to {config.dtype}")

    # Final verification
    model_device = next(model.parameters()).device
    print(f"Final model device: {model_device}")
    
    print(model)

    model = torch.compile(model, mode="reduce-overhead")
    eff_eval(model, tokenizer)