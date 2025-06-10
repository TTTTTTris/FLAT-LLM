#!/bin/bash

# Set common variables
model=meta-llama/Llama-2-70b-hf
tol=0.96
cuda_device=0,1,2,3
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
    --save_model $save_path \
    --seed 42
}

for sparsity_ratio in ...
do
    save_path="your path/out_models/llama-2-70b/Llama-2-70b-hf_$sparsity_ratio"
    mkdir -p "$save_path"  
    echo "Running with flatllm pruning method"
    run_python_command "flatllm" $sparsity_ratio "out/llama_70b/structured/" $tol 
    echo "Finished flatllm pruning method"
done