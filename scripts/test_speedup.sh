export CUDA_VISIBLE_DEVICES=5
dataset=wikitext2

###### Original ######
python eval.py \
        --model meta-llama/Llama-2-13b-hf \
        --device cuda:0 \
        --eval-dataset $dataset 2>&1 | tee -a test_speedup.txt

###### FLAT-LLM ######
for ratio in 88 76 64 52 40 # 88 77 66 55 44 # 90 80 70 60 50
do
python eval.py \
        --model meta-llama/Llama-2-13b-hf \
        --sliced-model-path ./out_models/llama-2-13b/no_QK_alpaca/Llama-2-13b-hf_${ratio} \
        --device cuda:0 \
        --eval-dataset $dataset 2>&1 | tee -a test_speedup.txt
done

###### SliceGPT ######
# for ratio in 0.2 0.3 0.4 0.45 0.55
# do
# python eval.py \
#         --model meta-llama/Llama-2-7b-hf \
#         --sliced-model-path ./out_slicegpt/llama-2-7b/$dataset/Llama-2-7b-hf_$ratio \
#         --device cuda:0 \
#         --eval-dataset $dataset 2>&1 | tee -a log_2.txt
# done

###### SVD-LLM ######
# for sparsity in 90 80 70 60 50
# do
# python eval.py \
#         --model meta-llama/Llama-2-7b-hf \
#         --sliced-model-path ./out_svdllm/checkpoints/svd_llm_llama_2_7b_$sparsity/model.pt \
#         --device cuda:0 \
#         --eval-dataset $dataset
# done
