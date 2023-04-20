from transformers import WhisperFeatureExtractor
from datasets import load_dataset
from transformers import WhisperTokenizer
from datasets import Audio

def prepare_dataset(batch, tokenizer, feature_extractor):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

def main():
    dataset = load_dataset("Bingsu/zeroth-korean")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="ko", task="transcribe")

    mapped_dataset = dataset.map(lambda x: prepare_dataset(x, tokenizer, feature_extractor), remove_columns=dataset.column_names["train"], num_proc=16)
    mapped_dataset.save_to_disk('./data/Bingsu_zeroth-korean', num_proc=4)

if __name__ == '__main__':
    main()