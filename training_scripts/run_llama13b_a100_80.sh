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


torchrun --nproc_per_node=8 code_trainer.py \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --model_revision 533ac5fc570d52216e713201835b7a3a2af990eb \
    --refactored_base_path TODO \
    --refactored_style $refactored_style \
    --final_style $final_style \
    --low_cpu_mem_usage True \
    --use_xformer_attn True \
    --bf16 True \
    --tf32 True \
    --output_dir "checkpoints_codellama_13b_apps_${short_refactored_style}_${short_final_style}_2e5_256_${num_epochs}" \
    --num_train_epochs $num_epochs \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
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
    --run_name "codellama_apps_13b_${short_refactored_style}_${short_final_style}_2e5_256_${$num_epochs}" \
    --do_train \
    --do_eval \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --final_style modularize_original \