#!/bin/bash

# Set common variables
model=mistralai/Mistral-7B-v0.1
bi_score='ranks/wikitext2/mistral-7b/'
tol=0.96
cuda_device=1
dataset=wikitext2

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

run_python_command () {
    python main_mistral.py \
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

for sparsity_ratio in 88 # 77 66 55 44
do
    save_path="your path/out_models/mistral-7b/Mistral-7b_$sparsity_ratio"
    mkdir -p "$save_path"  
    echo "Running with flatllm pruning method"
    run_python_command "flatllm" $sparsity_ratio "out/mistral_7b/structured/" $tol $save_path
    echo "Finished flatllm pruning method"
done