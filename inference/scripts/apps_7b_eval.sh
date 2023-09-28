difficulty=$1
model_path=$2
exp_name=$3
temperature=$4
num_samples=50
batch_size=50

if [ $difficulty == "intro" ]
then
    task="apps-introductory-cfstyle"
elif [ $difficulty == "interview" ]
then
    task="apps-interview-cfstyle"
elif [ $difficulty == "competition" ]
then
    task="apps-competition-cfstyle"
else
    echo "Invalid difficulty: $difficulty"
    exit 1
fi

if [ $temperature == "0.1" ]
then
    num_samples=15
    batch_size=15
fi

echo "Running inference on $task with $model_path and $temperature for $num_samples samples and $batch_size batch."

python main.py \
--model $model_path --use_auth_token \
--trust_remote_code --tasks $task --batch_size $batch_size --n_samples $num_samples \
--max_sequence_length 2048 --precision bf16 --temperature $temperature \
--exp_name "${exp_name}_7b_${difficulty}_${temperature}" --allow_code_execution --shuffle --num_gpus 8 \