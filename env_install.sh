# This script automates the complete setup for the NeuraTwin environment.
# It should be run from the root directory of the NeuraTwin project.
#
# Usage:
# Make sure CUDA-12.1 is installed and visible on your device.
# cd /path/to/NeuraTwin
# conda create -n NeuraTwin python=3.10 -y
# conda activate NeuraTwin
# bash env_install.sh

echo "Installing PhysTwin packages (0/3)"
bash env_install_phys.sh
echo "Installed Hanxiao's PhysTwin packages (1/3)"

pip install urdfpy
pip install --upgrade "networkx>=3.0"
pip install sapien
pip install einops
pip install plyfile
echo "Installed all PhysTwin packages (2/3)"

pip install gdown
pip install tensorboard
pip install lightning

pip install xarm-python-sdk
pip install timm
pip install transformers
pip install yapf
pip install pycocotools
pip install --force-reinstall charset_normalizer
echo "Installed all packages (3/3)"