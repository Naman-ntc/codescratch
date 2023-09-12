sudo apt install htop tmux unzip vim nano

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc
echo "export CPATH=/usr/local/cuda/include:$CPATH" >> ~/.bashrc
echo "export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export WANDB_PROJECT=CodeGenData" >> ~/.bashrc

source ~/.bashrc

pip install -r requirements.txt
pip install -U git+https://github.com/microsoft/DeepSpeed.git@jomayeri/issue-4095
wandb login 210e4d0a8c5f02a0da3661523868f6199b74c503

read -p "Install vllm? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    pip install git+https://github.com/vllm-project/vllm.git
fi