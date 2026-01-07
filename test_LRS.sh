#!/bin/bash

################################
#        GPU 轮询逻辑
################################
THRESHOLD=20000        # 20GB (MB)
CHECK_INTERVAL=300     # 5 分钟

echo "Waiting for GPU with >= ${THRESHOLD} MB free memory..."

while true; do
    FREE=$(nvidia-smi \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits | head -n 1)

    echo "[`date`] GPU Free Memory: ${FREE} MB"

    if [ "$FREE" -ge "$THRESHOLD" ]; then
        echo "✅ GPU is ready (Free: ${FREE} MB)"
        break
    fi

    echo "⏳ Not enough memory, retrying in ${CHECK_INTERVAL}s..."
    sleep ${CHECK_INTERVAL}
done

MODELS=("llama2-7b")
DATASETS=("cnn_dm_summarization")
SAMPLES=(1000)
SHOTS=(1)
LRS=(0.18 0.17 0.16 0.165 0.185)

file="test_LRS.py"
echo "Running Dynamic Skip Inference in ${file}"

infer_strategy="Dynamic_Skip_Inf"   

echo "Running Skip Args Test: ${file}"

for MODEL in "${MODELS[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    for SAMPLE in "${SAMPLES[@]}"; do
      for SHOT in "${SHOTS[@]}"; do
        for TH in "${LRS[@]}"; do
            echo " MODEL=$MODEL DATA=$DATA SAMPLE=$SAMPLE SHOT=$SHOT TH=$TH"
            python "$file" \
                --model_name "$MODEL" \
                --dataset_name "$DATA" \
                --samples "$SAMPLE" \
                --shots "$SHOT" \
                --infer_strategy "$infer_strategy" \
                --LRS_threshold "$TH" \
        
        done
      done
    done
  done
done