CUDA_VISIBLE_DEVICES=0 python evaluate_ai_hub_per_db.py \
    --noisy_root /data/ai_hub/synthesized_speech_with_noise_eval/NoisySpeech_training/ \
    --out_path ./data/out_eval.pckl

CUDA_VISIBLE_DEVICES=0 python evaluate_ai_hub_per_db.py \
    --noisy_root /data/ai_hub/synthesized_speech_with_noise_eval/NoisySpeech_training/ \
    --out_path ./data/out_eval_original.pckl \
    --processor_address openai/whisper-medium \
    --model_address openai/whisper-medium
    
CUDA_VISIBLE_DEVICES=0 python evaluate_ai_hub_per_db.py \
    --out_path ./data/out_eval_denoised.pckl

CUDA_VISIBLE_DEVICES=0 python evaluate_ai_hub_per_db.py \
    --out_path ./data/out_eval_denoised_original.pckl \
    --processor_address openai/whisper-medium \
    --model_address openai/whisper-medium
