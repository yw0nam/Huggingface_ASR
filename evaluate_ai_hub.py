#%%
import unicodedata
from pykospacing import Spacing
import os
import torch
import pandas as pd
import librosa
import re, evaluate
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm
from datasets import Dataset
from utils import remove_repeated_text
from torch.utils.data import DataLoader
# %%
spacing = Spacing()
# Read data and preprocessing 
csv = pd.read_csv("/data/ai_hub/AI-hub_korean_speech/scripts/eval_clean.trn", sep='::', names=['wav_path', 'text'])
csv['wav_path'] = csv['wav_path'].map(lambda x: os.path.join('/data/ai_hub/synthesized_speech_with_noise_eval/CleanSpeech_training/', os.path.basename(x)).replace('pcm', 'wav').rstrip())
csv['wav_length'] = csv['wav_path'].map(lambda x: len(librosa.load(x, sr=16000)[0]) / 16000)
csv = csv.query('wav_length > {}'.format(1.5)) # Clip by audio length
csv['text'] = csv['text'].map(lambda x: remove_repeated_text(x))

# Make dictionary for each snrdb and collate in one dataset
dataset = Dataset.from_pandas(csv, split='eval', preserve_index=False)
# %%
# Build processor and model
processor = WhisperProcessor.from_pretrained('openai/whisper-medium', language="ko", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained('./models_zoo/openai_whisper-medium/model_weights/').cuda()

# %%
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
# %%
def map_to_pred(batch, sr=16000):
    audio, _ = librosa.load(batch['wav_path'].strip(), sr=sr)
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
    
    # encode target text to label ids     
    text = unicodedata.normalize("NFKC",processor.tokenizer._normalize(batch['text']))
    batch["reference"] = spacing(text.replace(" ", ""))
    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = unicodedata.normalize("NFKC", processor.tokenizer._normalize(processor.decode(predicted_ids)))
    batch["prediction"] = spacing(transcription.replace(" ", ""))
    return batch

# Calculate predition
res = dataset.map(map_to_pred)
# %%
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
# %%
print(100 * wer_metric.compute(references=res['reference'], predictions=res['prediction']))
print(100 * cer_metric.compute(references=res['reference'], predictions=res['prediction']))
# %%
