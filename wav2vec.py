# %%
import unicodedata
import soundfile 
import os, librosa
import numpy as np
from pyctcdecode import build_ctcdecoder
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoTokenizer,
    Wav2Vec2ProcessorWithLM,
)
from transformers.pipelines import AutomaticSpeechRecognitionPipeline

# %%
model = AutoModelForCTC.from_pretrained("42MARU/ko-spelling-wav2vec2-conformer-del-1s")
feature_extractor = AutoFeatureExtractor.from_pretrained("42MARU/ko-spelling-wav2vec2-conformer-del-1s")
tokenizer = AutoTokenizer.from_pretrained("42MARU/ko-spelling-wav2vec2-conformer-del-1s")
beamsearch_decoder = build_ctcdecoder(
    labels=list(tokenizer.encoder.keys()),
    kenlm_model_path=None,
)
processor = Wav2Vec2ProcessorWithLM(
    feature_extractor=feature_extractor, tokenizer=tokenizer, decoder=beamsearch_decoder
)
# %%
asr_pipeline = AutomaticSpeechRecognitionPipeline(
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    decoder=processor.decoder,
    device=-1,
)
# %%
data_root = "/home/spow12/datas/2023_2nd_quarter/ASR/AI-hub_korean_speech/eval_data/eval_clean/"
data_name = "KsponSpeech_E00004.pcm"
data_path = os.path.join(data_root, data_name)
# %%
data_type = np.dtype('i2')  # 16-bit signed integer
data_size = os.path.getsize(data_path)  # get the size of the file in bytes
data_len = data_size // data_type.itemsize  # get the number of samples in the file

# pad the file with zeros if necessary
if data_size % data_type.itemsize != 0:
    pad_size = data_type.itemsize - data_size % data_type.itemsize
    with open(data_path, 'ab') as f:
        f.write(b'\x00' * pad_size)
# %%
# create a memory map for the file
data = np.memmap(data_path, dtype=data_type, mode='r', shape=(data_len,))
# %%
soundfile.write('KsponSpeech_E00004.wav', data, 16000)
# %%
raw_data, _ = librosa.load('KsponSpeech_E00004.wav', sr=16000)
kwargs = {"decoder_kwargs": {"beam_width": 100}}
# %%
pred = asr_pipeline(inputs=raw_data, **kwargs)["text"]
# 모델이 자소 분리 유니코드 텍스트로 나오므로, 일반 String으로 변환해줄 필요가 있습니다.
result = unicodedata.normalize("NFC", pred)
print(result)
# %%
