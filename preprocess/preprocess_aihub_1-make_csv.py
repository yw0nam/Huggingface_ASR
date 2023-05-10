# %%
import pandas as pd
from glob import glob
import os
import librosa, argparse
from tqdm import tqdm, tqdm_pandas
import sys

def define_argparser():
    """Function to define the command line arguments
    
    Returns:
        argparse.Namespace: Command line arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', type=str, default='/data/ai_hub/AI-hub_korean_speech/dev/')
    p.add_argument('--out_csv', type=str, default='./data/AI-hub_korean_speech.csv')
    config = p.parse_args()
    return config

def map_fn(aud_path, sr=16000):
    txt_path = aud_path.replace('wav', 'txt')
    if not os.path.exists(txt_path):
        return None, None
    with open(txt_path, 'r', encoding='cp949') as f:
        text = f.readline().rstrip()
    aud, _ = librosa.load(aud_path, sr=sr)
    aud_length = len(aud) / 16000
    return aud_length, text

def main(config):
    data_pathes = sorted(glob(os.path.join(config.root_dir, '*/*/*.wav')))
    df = pd.DataFrame({'audio_path' : data_pathes})
    
    tqdm.pandas()
    outs = df['audio_path'].progress_map(lambda x: map_fn(x))
    
    aud_length = [out[0] for out in outs]
    text = [out[1] for out in outs]
    
    df['text'] = text
    df['aud_length'] = aud_length
    df.to_csv(config.out_csv, index=False)
    
if __name__ == '__main__':
    config = define_argparser()
    main(config)
    