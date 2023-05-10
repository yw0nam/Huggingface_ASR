deepspeed --num_gpus=2 train_huggingface.py --deepspeed ./configs/deepspeed_stage_2.json

deepspeed --num_gpus=2 train_huggingface.py --deepspeed ./configs/deepspeed_stage_2.json --model_save_path ./models_zoo/openai_whisper-large-v2 --data_path /data/ai_hub/AI-hub_korean_speech/preprocessed/ --model_address openai/whisper-large-v2 --batch_size_per_device 16