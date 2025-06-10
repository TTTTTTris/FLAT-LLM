#!/bin/bash

# Set common variables
model=mistralai/Mistral-7B-v0.1
bi_score='ranks/wikitext2/mistral-7b/'
cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

run_python_command () {
    python main_mistral.py \
    --model $model \
    --prune_method $1 \
    --bi_score $bi_score \
    --sparsity_ratio $2 \
    --save $3 
}

for sparsity_ratio in 1
do
    echo "Running with BI score computation"
    run_python_command "bi" $sparsity_ratio "out/mistral_7b/structured/"
    echo "Finished BI score computation"
done