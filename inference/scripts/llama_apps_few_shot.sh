python main.py \
--model codellama/CodeLlama-7b-hf --use_auth_token \
--trust_remote_code --tasks appsfewshot-introductory-base --batch_size 10 --n_samples 50 \
--max_sequence_length 6154 --precision bf16 --temperature 0.1 \
--num_gpus 4 --exp_name clfs_intro_base_01 --allow_code_execution --shuffle --num_gpus 4 \