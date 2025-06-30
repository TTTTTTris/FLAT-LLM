export CUDA_VISIBLE_DEVICES=2
model=meta-llama/Llama-2-7b-hf


for ratio in 64
do
    export MODEL_PATH="your path/out_models/llama-2-7b/Llama-2-7b-hf_${ratio}"
    python quant_flatllm.py \
        --model $model \
        --model_path $MODEL_PATH \
        --dataset wikitext2 \
        --wbits 3 \
        --act-order \
        --new-eval  \
        --sparsity_ratio ${ratio} \
        --save "out/llama_7b/structured/" \
        --save_model ${MODEL_PATH}_gptq
done