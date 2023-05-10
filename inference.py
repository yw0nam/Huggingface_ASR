import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import argparse

def define_argparser():
    """Function to define the command line arguments
    
    Returns:
        argparse.Namespace: Command line arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument('--wav_path', type=str, required=True)
    p.add_argument('--out_path', type=str, required=True)
    p.add_argument('--model_address', type=str, required=True)
    p.add_argument('--processor_address', type=str, required=True)
    p.add_argument('--gpu', type=str, default='cuda', choices=['cuda', 'cpu'])
    config = p.parse_args()
    return config

def main(config):
    processor = WhisperProcessor.from_pretrained(config.processor_address, language="ko", task="transcribe")
    data, _ = librosa.load(config.wav_path, sr=16000)
    
    if config.gpu == 'cuda':
        model = WhisperForConditionalGeneration.from_pretrained(config.model_address).cuda()
        input_features = processor(data, sampling_rate=16000, return_tensors="pt").input_features.cuda()
    else:
        model = WhisperForConditionalGeneration.from_pretrained(config.model_address)
        input_features = processor(data, sampling_rate=16000, return_tensors="pt").input_features
    
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    with open(config.out_path, 'w', encoding='UTF-8') as f:
        f.write(transcription[0])
        
if __name__ == '__main__':
    config = define_argparser()
    main(config)