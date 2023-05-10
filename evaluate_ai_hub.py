#%%
import os
import torch
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
import re, evaluate
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from utils import DataCollatorSpeechSeq2SeqWithPadding_from_npy
from torch.utils.data import DataLoader
# %%
csv = pd.read_csv('/data/ai_hub/AI-hub_korean_speech/scripts/eval_clean.trn', sep='::', names=['wav_path', 'text'])
# %%
root_path = "/data/ai_hub/AI-hub_korean_speech/eval_data/eval_clean"
csv['wav_path'] = csv['wav_path'].map(lambda x: os.path.join(root_path, os.path.basename(x)).replace('pcm', 'wav'))
# %%
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
# %%
def remove_repeated_text(text, comp):
    """ remove repeated text 
    Here is example
    Input:  'o/ b/ 그게 (0.1프로)/(영 점 일 프로) 가정의 아이들과 가정의 모습이야? b/'
    Output: '그게 0.1프로 가정의 아이들과 가정의 모습이야?'
    Args:
        text (str): string from pandas Series
    Returns:
        text (str): Removed text 
    """
    matched_text = comp.search(text)
    if matched_text:
        text = text.replace(matched_text[0], re.sub('\(|\)', "", matched_text[0].split('/')[0]))
    text = re.sub("o/|c/|n/|N/|u/|l/|b/|\*|\+|/", " ", text)
    
    return text.rstrip()
comp = re.compile('\(.*\)/\(.*\)')
# %%
csv['text'] = csv['text'].map(lambda x: remove_repeated_text(x, comp))
# %%
ds = Dataset.from_pandas(csv)
# %%
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="ko", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained('./models_zoo/openai_whisper-large-v2/model_weights/').cuda()
# %%
def map_to_pred(batch, sr=16000):
    audio, _ = librosa.load(batch['wav_path'].strip(), sr=sr)
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
    
    # encode target text to label ids     
    batch["reference"] = processor.tokenizer._normalize(batch['text'])
    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    return batch
# %%
res = ds.map(map_to_pred)
# %%
print(100 * wer_metric.compute(references=res["reference"], predictions=res["prediction"]))
print(100 * cer_metric.compute(references=res["reference"], predictions=res["prediction"]))
# whisper-medium WER : 21.44501150805564
# whisper-medium CER : 6.681410778721918
# %%
# whisper-large_V2 WER : 20.699489642749924
# whisper-large_v2 CER : 6.362472354789897