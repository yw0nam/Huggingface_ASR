# CUDA_VISIBLE_DEVICES=0 python evaluate_ai_hub_per_db.py \
#     --noisy_root /data/ai_hub/synthesized_speech_with_noise_eval/NoisySpeech_training/ \
#     --out_path ./data/out_eval.pckl

# CUDA_VISIBLE_DEVICES=0 python evaluate_ai_hub_per_db.py \
#     --noisy_root /data/ai_hub/synthesized_speech_with_noise_eval/NoisySpeech_training/ \
#     --out_path ./data/out_eval_original.pckl \
#     --processor_address openai/whisper-medium \
#     --model_address openai/whisper-medium
    
# CUDA_VISIBLE_DEVICES=0 python evaluate_ai_hub_per_db.py \
#     --out_path ./data/out_eval_denoised.pckl

# CUDA_VISIBLE_DEVICES=0 python evaluate_ai_hub_per_db.py \
#     --out_path ./data/out_eval_denoised_original.pckl \
#     --processor_address openai/whisper-medium \
#     --model_address openai/whisper-medium

# CUDA_VISIBLE_DEVICES=0 python evaluate_ai_hub_per_db.py \
#     --out_path ./data/out_eval_fintunned-with-denoised_denoised.pckl \
#     --model_address /home/spow12/codes/2023_Q2/ASR/models_zoo/openai_whisper_medium_denoise/model_weights/

# CUDA_VISIBLE_DEVICES=1 python evaluate_ai_hub_per_db.py \
#     --out_path ./data/out_eval_fintunned-with-noise_noise.pckl \
#     --model_address /home/spow12/codes/2023_Q2/ASR/models_zoo/openai_whisper_medium_noise/model_weights \
#     --noisy_root /data/ai_hub/synthesized_speech_with_noise_eval/NoisySpeech_training/

CUDA_VISIBLE_DEVICES=1 python evaluate_ai_hub_per_db.py \
    --out_path ./data/out_eval_fintunned-with-noise_speech_noise.pckl \
    --model_address /home/spow12/codes/2023_Q2/ASR/models_zoo/openai_whisper_medium_noise/model_weights \
    --noisy_root /data/ai_hub/synthesized_speech_with_speech_noise_eval/NoisySpeech_training/