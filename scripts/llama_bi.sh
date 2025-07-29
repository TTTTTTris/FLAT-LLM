#!/bin/bash

# Set common variables
model=meta-llama/Meta-Llama-3-8B
bi_score='ranks/wikitext2/llama-3-8b/'
cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --bi_score $bi_score \
    --sparsity_ratio $2 \
    --save $3 
}

for sparsity_ratio in 1
do
    echo "Running with BI score computation"
    run_python_command "bi" $sparsity_ratio "out/llama_3_8b/structured/"
    echo "Finished BI score computation"
done