export CUDA_VISIBLE_DEVICES=2,3
python demo_audiovideo.py \
    --cfg-path eval_configs/video_llama_eval_withaudio.yaml \
    --model_type llama_v2 \
    --gpu-id 0