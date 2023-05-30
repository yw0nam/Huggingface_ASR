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

e. Install DeepSpeed

First you need libaio-dev. please install by

```shell
sudo apt-get install libaio-dev
```

After this, install deepspeed by 

```shell
DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_UTILS=1 DS_BUILD_AIO=1 pip install deepspeed==0.9.0 --global-option="build_ext" --global-option="-j11" --no-cache-dir
```

For detail instruction for installing Deepspeed, Check [official site](https://github.com/microsoft/DeepSpeed)

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

