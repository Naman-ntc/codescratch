difficulty=$1
model_path=$2
exp_name=$3
temperature=$4

if [ $difficulty == "easy" ]
then
    task="codecontests-easy"
elif [ $difficulty == "medium" ]
then
    task="codecontests-medium"
elif [ $difficulty == "hard" ]
then
    task="codecontests-hard"
elif [ $difficulty == "test" ]
then
    task="codecontests-test"
else
    echo "Invalid difficulty: $difficulty"
    exit 1
fi


python main.py \
--model $model_path --use_auth_token \
--trust_remote_code --tasks $task --batch_size 50 --n_samples 200 \
--max_sequence_length 2048 --precision bf16 --temperature $temperature \
--exp_name "${exp_name}_7b_${difficulty}_${temperature}" --allow_code_execution --shuffle --num_gpus 4 \