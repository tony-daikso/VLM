set -x
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv venv --python 3.12 ~/Tony/venv
. ~/Tony/venv/bin/activate
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install numpy==2.2.6 scipy==1.16.3 pandas==3.0.6 scikit-learn==1.9.1 h5py pyyaml timm==1.0.22 nnunetv2==2.8.1 TotalSegmentator==2.18.0 dynamic_network_architectures==0.4.4 batchgenerators==0.25.5 nibabel simpleitk
python -c "import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.device_count()); x=torch.randn(1000,1000,device=\"cuda\"); print((x@x).sum().item())"
echo SETUP_DONE
