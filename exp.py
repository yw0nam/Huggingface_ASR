#%%
from sklearn.model_selection import train_test_split
import shutil
import os
import pandas as pd
import evaluate
import pickle
from glob import glob
# %%
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer") 
# %%
pckl_path = glob('./data/out_eval_fintunned-with-noise_speech_noise.pckl')
# %%
df_ls = []
for path in pckl_path:
    with open(path, 'rb') as f:
        eval = pickle.load(f)
    dicts = {}
    for snr_db in eval.keys():
        dicts[snr_db] = {
            'WER' : 100 * wer_metric.compute(references=eval[snr_db]["reference"], predictions=eval[snr_db]["prediction"]),
            'CER' : 100 * cer_metric.compute(references=eval[snr_db]["reference"], predictions=eval[snr_db]["prediction"])
        }
    df = pd.DataFrame(dicts)
    df['model'] = os.path.basename(path[:-5])
    df_ls.append(df)
# %%
final_df = pd.concat(df_ls)
# %%
final_df
# %%
eval
# %%
shutil.copy(eval['SNRdb_10.0']['wav_path'][0], './')

# %%
ref = eval['SNRdb_10.0']['reference'][:2]
pred = eval['SNRdb_10.0']['prediction'][:2]
# %%
wer_metric.compute(references=ref, predictions=pred)
# %%
csv = pd.read_csv('./data/AI-hub_korean_speech_noise.csv')
# %%
train, val = train_test_split(csv, test_size=5000, stratify=csv['aud_length'].astype(int), random_state=1004)
# %%
train.to_csv('./temp/train.csv',index=False)
val.to_csv('./temp/val.csv',index=False)
# %%
