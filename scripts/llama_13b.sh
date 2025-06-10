#!/bin/bash

# Set common variables
model=meta-llama/Llama-2-13b-hf
bi_score='ranks/wikitext2/llama-2-13b/'
tol=0.96
cuda_device=0
dataset=wikitext2

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $2 \
    --dataset $dataset \
    --bi_score $bi_score \
    --save $3 \
    --tol $4 \
    --nsamples 256 \
    --save_model $5 \
    --seed 42
}

for sparsity_ratio in 75 62 50 38
do
    save_path="your path/out_models/llama-2-13b/Llama-2-13b-hf_$sparsity_ratio"
    mkdir -p "$save_path"  
    echo "Running with flatllm pruning method"
    run_python_command "flatllm" $sparsity_ratio "out/llama_13b/structured/" $tol $save_path
    echo "Finished flatllm pruning method"
done