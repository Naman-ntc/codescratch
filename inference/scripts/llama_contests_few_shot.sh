python main.py \
--model codellama/CodeLlama-7b-hf --use_auth_token \
--trust_remote_code --tasks codecontestsfewshot-easy-base --batch_size 10 --n_samples 50 \
--max_sequence_length 6154 --precision bf16 --temperature 0.7 --max_num_batched_tokens 16000 \
--exp_name clfs_contests_easy_base_07 --allow_code_execution --shuffle \