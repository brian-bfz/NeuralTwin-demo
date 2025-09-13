NeuralTwin is a pipeline for transferring physics simulators of soft objects into neural networks, enabling 1000x faster robotic applications. It uses the model's uncertainty as a reward signal to autonomously explore the simulation. We plan to open source NeuralTwin soon. This codebase is a simplified demo, containing a 700k param transformer trained on 3 hours of rope data with NeuralTwin. [Video](https://drive.google.com/file/d/1noDw_VjRvOF77Gff8grztGzQK6CRGdPI/view?usp=sharing)

## Install
Install CUDA-12.1 and add it to your path

```
conda create -n NeuraTwin python=3.10 -y
conda activate NeuraTwin
bash env_install.sh
```

Download [data.zip](https://drive.google.com/file/d/1RkyE417ZKRnJ801gig7o7UhK6wKucxDX/view?usp=sharing), decompress it, and put it in the root directory 

## Usage
```python interactive.py --model m0 --motion [push or lift]```

This should create an interactive window, in which you can use your keyboard (WASD, QE) to manipulate a rope. It uses a transformer pretrained with NeuralTwin to simulate how the rope would deform. 
