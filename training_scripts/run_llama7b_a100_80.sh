torchrun --nproc_per_node=8 --rdzv-endpoint localhost:29512 apps_monkeypatch_trainer.py \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --model_revision 533ac5fc570d52216e713201835b7a3a2af990eb \
    --refactored_base_path /root/program_refactoring/logs/refactoring \
    --refactored_style base_original \
    --low_cpu_mem_usage True \
    --use_xformer_attn True \
    --bf16 True \
    --tf32 True \
    --output_dir checkpoints_codellama_7b_base_4e5_256_4 \
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
    --learning_rate 4e-5 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --block_size 2048 \
    --report_to wandb \
    --run_name codellama_7b_base_4e5_256_4 \
    --do_train \
    --do_eval \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --final_style modularize_original \