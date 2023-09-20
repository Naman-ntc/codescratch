# export NCCL_P2P_DISABLE=1
declare -A dictionary
dictionary=(
    ['base']='base_original'
    ['remod']='remodularize_merged'
)

short_refactored_style=$1

refactored_style=${dictionary[$short_refactored_style]}

if [ -z "$2" ]
then
    num_epochs=2
else
    num_epochs=$2
fi


deepspeed code_trainer.py \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --refactored_base_path /root/codescratch/apps_enumerated_old/ \
    --oai_mode \
    --refactored_style $refactored_style \
    --low_cpu_mem_usage True \
    --use_xformer_attn True \
    --bf16 True \
    --tf32 True \
    --output_dir "checkpoints_codellama_7b_appsoai_${short_refactored_style}_5e5_256_${num_epochs}" \
    --num_train_epochs $num_epochs \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 8 \
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
    --run_name "codellama_7b_appsoai_${short_refactored_style}_5e5_256_${num_epochs}" \
    --do_train \
    --do_eval \
    --deepspeed utils/ds_config_cpu.json \
    # --final_style modularize_original \