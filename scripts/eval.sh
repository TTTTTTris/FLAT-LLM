
export CUDA_VISIBLE_DEVICES=5
model=meta-llama/Meta-Llama-3-8B

# for ratio in 80
# do
# export MODEL_PATH="./out_models/llama-3-8b/Llama-3-8b-hf_${ratio}/pytorch_model.bin"

#    python -m lm_eval \
#       --model hf \
#       --model_args pretrained=${model},parallelize=True \
#       --tasks piqa,hellaswag,arc_easy,arc_challenge,winogrande,boolq,openbookqa \
#       --wandb_args project=lm-eval-harness-integration \
#       --output_path ./eval_results \
#       --batch_size 1
#       # --tasks piqa,winogrande,hellaswag,arc_easy,arc_challenge,openbookqa \
# done

for ratio in 80 
do
export MODEL_PATH="./out_models/llama-3-8b/Llama-3-8b-hf_${ratio}/pytorch_model.bin"
   python -m lm_eval \
      --model hf \
      --model_args pretrained=${model},parallelize=True \
      --tasks mmlu \
      --num_fewshot 5 \
      --wandb_args project=lm-eval-harness-integration \
      --output_path ./eval_results \
      --batch_size 1
done

# hellaswag,arc_easy,arc_challenge,winogrande,boolq,piqa,openbookqa 
# super-glue-lm-eval-v1,openbookqa,truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen \
# mrpc,hellaswag,arc_easy,arc_challenge,winogrande,boolq,rte,piqa,mathqa,commonsense_qa,mmlu \
# wic,mrpc,hellaswag,arc_easy,arc_challenge,winogrande,boolq,rte,
