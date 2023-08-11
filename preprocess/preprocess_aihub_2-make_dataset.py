import pandas as pd
from sklearn.model_selection import train_test_split
import argparse, os
from datasets import Dataset, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import sys
from tqdm import tqdm
import gc

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from utils import prepare_dataset_aihub, remove_repeated_text

def define_argparser():
    """
    Define command line argument parser.
    
    Returns:
        argparse.Namespace: Command line arguments
    """
    p = argparse.ArgumentParser()

    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--val_size", type=int, default=5000)
    p.add_argument("--clip_length", type=float, default=1.5)
    p.add_argument("--model_address", type=str, default="openai/whisper-medium")
    config = p.parse_args()

    return config

def main(config):
    
    csv = pd.read_csv(config.csv_path)
    csv = csv.query("aud_length > {}".format(config.clip_length))
    tqdm.pandas()
    csv['text'] = csv['text'].progress_map(lambda x: remove_repeated_text(x))
    train, val = train_test_split(csv, test_size=config.val_size, stratify=csv['aud_length'].astype(int), random_state=1004)
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model_address)
    tokenizer = WhisperTokenizer.from_pretrained(config.model_address, language="ko", task="transcribe")
    
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train, split='train', preserve_index=False),
        'val': Dataset.from_pandas(val, split='val', preserve_index=False),
    })
    
    del (csv, train, val)
    gc.collect()
    dataset = dataset.remove_columns(['aud_length'])
    mapped_dataset = dataset.map(lambda x: prepare_dataset_aihub(x, tokenizer, feature_extractor),
                                remove_columns=dataset.column_names["train"], 
                                num_proc=8,
                                )
    mapped_dataset.save_to_disk(config.out_path, num_proc=8)

if __name__ == '__main__':
    config = define_argparser()
    main(config)

