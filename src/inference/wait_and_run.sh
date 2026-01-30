#!/bin/bash
# Wait for GPU memory to free up, then run v8 production inference (all 1000 MMAR samples).
# Usage: nohup bash wait_and_run.sh &

REQUIRED_FREE_MB=70000  # 70GB free
CHECK_INTERVAL=60       # check every 60 seconds
GPU_ID=0

echo "$(date): Waiting for ${REQUIRED_FREE_MB}MB free on GPU ${GPU_ID}..."
echo "Checking every ${CHECK_INTERVAL}s. Run 'tail -f wait_and_run.log' to monitor."

while true; do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_ID | tr -d ' ')

    if [ "$FREE_MB" -ge "$REQUIRED_FREE_MB" ]; then
        echo "$(date): GPU ${GPU_ID} has ${FREE_MB}MB free (>= ${REQUIRED_FREE_MB}MB). Launching inference..."
        break
    else
        echo "$(date): GPU ${GPU_ID} has ${FREE_MB}MB free. Waiting..."
        sleep $CHECK_INTERVAL
    fi
done

cd /home/ikulkar1/qwen_omni_finetune/audio_reasoning_interspeech/src/inference

time python infer_single_model_finetuned_v8.py \
    --dataset_meta_path ../data/MMAR-meta.json \
    --dataset_audio_prefix ../data \
    --qwen3_omni_model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --adapter_path ../models/rest_Qwen_Qwen3_Omni_30B_A3B_Thinking_20260125_210524 \
    --output_dir ../../outputs/v8_finetuned \
    --batch_size 1 \
    --do_sample False \
    --resume

echo "$(date): Inference complete."
