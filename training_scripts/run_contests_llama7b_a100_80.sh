export MODEL_DIR="codellama"
declare -A dictionary
dictionary=(
    ['base']='base_original'
    ['rename']='rename_original'
    ['mod']='modularize_original'
    ['remod']='remodularize_merged'
    ['remod_35']='remodularize_merged_35'
    ['planm1']='plan_merged1'
    ['planm2']='plan_merged2'
    ['planm1padall']='plan_merged1padall'
    ['planm2padall']='plan_merged2padall'
    ['planm1pad2']='plan_merged1pad2'
    ['planm2pad2']='plan_merged2pad2'
    ['planm1pad1']='plan_merged1pad1'
    ['planm2pad1']='plan_merged2pad1'
    ['planm1maskall']='plan_merged1maskall'
    ['planm2maskall']='plan_merged2maskall'
    ['planm1mask2']='plan_merged1mask2'
    ['planm2mask2']='plan_merged2mask2'
)


short_refactored_style=$1
short_final_style=$2

if [ -z "$3"]
then
    lr=5e-5
else
    lr=$4
fi

refactored_style=${dictionary[$short_refactored_style]}
final_style=${dictionary[$short_final_style]}

deepspeed code_trainer.py \
    --model_name_or_path $MODEL_DIR/CodeLlama-7b-hf \
    --refactored_base_path code_contests_enumerated_train \
    --refactored_style $refactored_style \
    --final_style $final_style \
    --low_cpu_mem_usage True \
    --use_xformer_attn True \
    --bf16 True \
    --tf32 True \
    --output_dir "checkpoints_codellama_7b_contests_${short_refactored_style}_${short_final_style}_${lr}_256_1" \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --save_strategy "steps" \
    --save_steps 25 \
    --save_total_limit 10 \
    --evaluation_strategy "steps" \
    --eval_steps 25 \
    --learning_rate $lr \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --block_size 2048 \
    --report_to wandb \
    --run_name "codellama_7b_contests_${short_refactored_style}_${short_final_style}_${lr}_256_1" \
    --do_train \
    --do_eval \
    --deepspeed utils/ds_config.json \