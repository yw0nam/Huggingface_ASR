import unicodedata
from pykospacing import Spacing
import numpy as np
import os
import pickle
import torch
import pandas as pd
import librosa
import argparse
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import Dataset, DatasetDict
from utils import remove_repeated_text

def define_argparser():
    """Function to define the command line arguments
    
    Returns:
        argparse.Namespace: Command line arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument('--clean_root', type=str, default='/data/ai_hub/synthesized_speech_with_noise_eval/CleanSpeech_training/', 
                help="Clean speech data root for making reference wav pathes")
    p.add_argument('--noisy_root', type=str, default='/data/ai_hub/denoised_synthesized_speech_with_noise_eval/enhanced_0126/', 
                help='Noisy speech data root for reading the noisy wav')
    p.add_argument('--metadata_path', type=str, default='/data/ai_hub/AI-hub_korean_speech/scripts/eval_clean.trn',
                help='metadata path for parsing wav pathes')
    p.add_argument('--out_path', type=str, default='./data/out_eval_cliped.pckl',
                help='Dataset dict which has prediction and reference both')
    p.add_argument('--processor_address', type=str, default='openai/whisper-medium')
    p.add_argument('--model_address', type=str, default='spow12/whisper-medium-ksponspeech')
    p.add_argument('--clip_length', type=float, default=1.5)
    config = p.parse_args()
    return config

def main(args):
    """
    Main function for data processing and prediction.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        None
    """
    
    spacing = Spacing()
    # Read data and preprocessing 
    csv = pd.read_csv(args.metadata_path, sep='::', names=['wav_path', 'text'])
    csv['wav_path'] = csv['wav_path'].map(lambda x: os.path.join(args.clean_root, os.path.basename(x)).replace('pcm', 'wav').rstrip())
    csv['wav_length'] = csv['wav_path'].map(lambda x: len(librosa.load(x, sr=16000)[0]) / 16000)
    csv = csv.query('wav_length > {}'.format(args.clip_length)) # Clip by audio length
    csv['text'] = csv['text'].map(lambda x: remove_repeated_text(x))
    
    # Make dictionary for each snrdb and collate in one dataset
    snrs = np.linspace(-40, 40, 17)
    dataset_dict = {}
    for snr in snrs:
        temp_csv = csv.copy()
        key = 'SNRdb_{}'.format(str(snr))
        temp_csv['wav_path'] = csv['wav_path'].map(lambda x: os.path.join(
            args.noisy_root, '{}'.format(os.path.basename(x)[:-4]) + '_SNRdb_'+str(snr)+'.wav'))
        dataset_dict[key] = Dataset.from_pandas(temp_csv, split=key, preserve_index=False)
    
    dataset = DatasetDict(dataset_dict)
    
    # Build processor and model
    processor = WhisperProcessor.from_pretrained(args.processor_address, language="ko", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_address).cuda()
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    
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
    
    # Store results
    with open(args.out_path, 'wb') as f:
        pickle.dump(res, f)
        
if __name__ == '__main__':
    config = define_argparser()
    main(config)