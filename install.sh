pip install -r requirements.txt

read -p "Install vllm? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    pip install git+https://github.com/vllm-project/vllm.git
fi