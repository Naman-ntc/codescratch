# export NCCL_P2P_DISABLE=1
declare -A dictionary
dictionary=(
    ['base']='base_original'
    ['rename']='rename_original'
    ['mod']='modularize_original'
    ['remod']='remodularize_merged'
    ['planm1']='plan_merged1'
    ['planm2']='plan_merged2'
    ['planm1padall']='plan_merged1padall'
    ['planm2padall']='plan_merged2padall'
    ['planm1pad2']='plan_merged1pad2'
    ['planm2pad2']='plan_merged2pad2'
)

short_refactored_style=$1
short_final_style=$2

refactored_style=${dictionary[$short_refactored_style]}
final_style=${dictionary[$short_final_style]}


deepspeed code_trainer.py \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --model_revision 533ac5fc570d52216e713201835b7a3a2af990eb \
    --refactored_base_path /root/codescratch/code_contests_enumerated_train/ \
    --refactored_style $refactored_style \
    --final_style $final_style \
    --low_cpu_mem_usage True \
    --use_xformer_attn True \
    --bf16 True \
    --tf32 True \
    --output_dir "checkpoints_codellama_7b_contests_${short_refactored_style}_${short_final_style}_5e5_256_1" \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --save_strategy "steps" \
    --save_steps 25 \
    --save_total_limit 10 \
    --evaluation_strategy "steps" \
    --eval_steps 25 \
    --learning_rate 5e-5 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --block_size 2048 \
    --report_to wandb \
    --run_name "codellama_7b_contests_${short_refactored_style}_${short_final_style}_5e5_256_1" \
    --do_train \
    --do_eval \
    --deepspeed utils/ds_config.json \
    # --final_style modularize_original \