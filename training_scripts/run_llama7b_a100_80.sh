refactored_style=$1
final_style=$2

dictionary = {
    'base_original': 'base',
    'rename_original': 'rename',
    'modularize_original': 'mod',
    'remodularize_merged': 'remod',
    'plan_merged1': 'planm1',
    'plan_merged2': 'planm2',
    'plan_merged1padall': 'planm1padall',
    'plan_merged2padall': 'planm2padall',
}

short_refactored_style=${dictionary[$refactored_style]}
short_final_style=${dictionary[$final_style]}

torchrun --nproc_per_node=8 --rdzv-endpoint localhost:29512 code_trainer.py \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --model_revision 533ac5fc570d52216e713201835b7a3a2af990eb \
    --refactored_base_path TODO \
    --refactored_style $refactored_style \
    --final_style $final_style \
    --low_cpu_mem_usage True \
    --use_xformer_attn True \
    --bf16 True \
    --tf32 True \
    --output_dir "checkpoints_codellama_7b_apps_${short_refactored_style}_2e5_256_2" \
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
    --learning_rate 2e-5 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --block_size 2048 \
    --report_to wandb \
    --run_name codellama_7b_apps_base_2e5_256_4 \
    --do_train \
    --do_eval \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --final_style modularize_original \