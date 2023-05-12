# Finetunning ASR models

I-BRICKS - Auto Speech Recognition using huggingface

# Quickstart

## Experiment Setting

- Linux
- Python 3.9
- PyTorch 2.0.0 and CUDA

a. Create a conda virtual environment and activate it.

```shell
conda create -n ASR python=3.9
conda activate ASR
```
b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/)

c. Clone this repository.

```shell
git clone https://github.com/yw0nam/Huggingface_ASR/
cd Huggingface_ASR
```

d. Install requirments.

```shell
pip install -r requirements.txt
```

# Train the model

Training the model using below code.

```
CUDA_VISIBLE_DEVICES=0,1 python train_huggingface.py
```

Check argument in train_huggingface.py for training detail 

# Inference

Try this

```
CUDA_VISIBLE_DEVICES=0 python inference.py --wav_path YOUR_WAV_PATH --out_path PATH_FOR_OUT_TEXT
```

