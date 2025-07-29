export CUDA_VISIBLE_DEVICES=5
model=meta-llama/Meta-Llama-3-8B

for wbits in 2 3 4
do
    export MODEL_PATH="your path/out_models/llama-2-7b/Llama-2-7b-hf_${ratio}"
python quant_flatllm.py \
    --model $model \
    --dataset wikitext2 \
    --wbits $wbits \
    --act-order \
    --new-eval  \
    --save "out/llama_3_8b/structured/" \
    --save_model ${MODEL_PATH}_gptq_${wbits}
done