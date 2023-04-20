# %%
import shutil
import os
import librosa
import pandas as pd
from glob import glob
import numpy as np
root_path = "/data/ai_hub/AI-hub_korean_speech/dev/"

# %%
pathes = glob(os.path.join(root_path, '*/*/*.wav'))
# %%
raw_data, _ = librosa.load(pathes[0], sr=16000)
# %%
len(raw_data) / 16000
# %%
shutil.copy(pathes[0], './')
# %%
with open(pathes[0].replace('.wav', '.txt'), 'r', encoding='cp949') as f:
    lines = f.readline().rstrip()
# %%
lines
# %%
