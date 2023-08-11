#%%
import os
import unicodedata
import torch
import pandas as pd
import librosa
import re, evaluate
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import Dataset, load_from_disk
from utils import remove_repeated_text
# %%
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
# %%
processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="ko", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained('./models_zoo/openai_whisper-medium/model_weights/').cpu()
# %%
ds = load_from_disk('//data/ai_hub/AI-hub_korean_speech/preprocessed/with_noise/')
# %%
def map_to_pred(batch, sr=16000):
    audio, _ = librosa.load(batch['new_wav_path'].strip(), sr=sr)
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
    
    # encode target text to label ids  
    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cpu"))[0]
    batch["prediction"] = unicodedata.normalize("NFKC", processor.tokenizer._normalize(processor.decode(predicted_ids)))
    return batch
# %%
res = ds.map(map_to_pred)
# %%
print(100 * wer_metric.compute(references=res["reference"], predictions=res["prediction"]))
print(100 * cer_metric.compute(references=res["reference"], predictions=res["prediction"]))
# %%

# %%
import unicodedata
csv = pd.DataFrame(res)
# %%
csv['prediction'] = csv['prediction'].map(lambda x: join_jamos(x))
csv['reference'] = csv['reference'].map(lambda x: join_jamos(x))
# %%
csv.to_excel('./temp.xlsx',index=False)
# %%
spacer.space([csv['reference'][0].replace(" ", "")])
# %%
unicodedata.normalize("NFKC", csv['prediction'][0])
# %%
with open('./temp.txt', 'w') as f:
    for wav_path, prediction in zip(res['new_wav_path'], res['prediction']):
        f.write(f"{wav_path}\t{prediction}\n")
# %%
