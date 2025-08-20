This is a temporary demo of NeuralTwin. NeuralTwin is a pipeline for transferring a physics simulator of deformable objects into a neural network. It uses the model's uncertainty as a reward to autonomously explore the simulation. We plan to open source NeuralTwin soon. Before then, recruiters can use this demo to review my technical skills. 

## Install
Install CUDA-12.1 and add it to your path

```conda create -n NeuraTwin python=3.10 -y```

```conda activate NeuraTwin```

```bash env_install.sh```

Download [data.zip](https://drive.google.com/file/d/1RkyE417ZKRnJ801gig7o7UhK6wKucxDX/view?usp=sharing), decompress it, and put it in the root directory 

## Usage
```python interactive.py --model m0 --motion [push or lift]```

This should create an interactive window, in which you can use your keyboard (WASD, QE) to manipulate a rope. It uses a transformer pretrained with NeuralTwin to simulate how the rope would deform. 