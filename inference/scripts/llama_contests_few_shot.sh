python main.py \
--model codellama/CodeLlama-7b-hf --use_auth_token \
--trust_remote_code --tasks humaneval --batch_size 10 --n_samples 50 \
--max_sequence_length 1024 --precision bf16 --temperature 0.7 --max_num_batched_tokens 16000 \
--exp_name starcoder_humaneval_03 --allow_code_execution --shuffle \