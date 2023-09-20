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
    ['planm1pad1']='plan_merged1pad1'
    ['planm2pad1']='plan_merged2pad1'
)

short_refactored_style=$1
short_final_style=$2

refactored_style=${dictionary[$short_refactored_style]}
final_style=${dictionary[$short_final_style]}

if [ -z "$3" ]
then
    num_epochs=2
else
    num_epochs=$3
fi

deepspeed code_trainer.py \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --refactored_base_path apps_enumerated_old \
    --refactored_style $refactored_style \
    --final_style $final_style \
    --low_cpu_mem_usage True \
    --use_xformer_attn True \
    --bf16 True \
    --tf32 True \
    --filter_on_passed False \
    --output_dir "checkpoints_codellama_7b_apps_${short_refactored_style}_${short_final_style}_5e5_256_${num_epochs}" \
    --num_train_epochs 2 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
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
    --run_name "codellama_7b_apps_${short_refactored_style}_${short_final_style}_5e5_256_${num_epochs}" \
    --do_train \
    --do_eval \
    --deepspeed utils/ds_config.json \