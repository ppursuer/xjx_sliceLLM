#!/bin/bash

#    切换到项目路径
cd ../

################################
#        GPU 轮询逻辑
################################
THRESHOLD=20000        # 20GB (MB)
CHECK_INTERVAL=600     # 10 分钟

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
SAMPLES=(-1)
SHOTS=(1)

######################################
#    Skip Inference 实验
######################################

file="exp_1_skip_update.py"
infer_strategy="Dynamic_Skip_Inf"   
LRS_TH=0.185
echo "MODEL=$MODEL DATA=$DATA SAMPLE=$SAMPLE SHOT=$SHOT"

for MODEL in "${MODELS[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    for SAMPLE in "${SAMPLES[@]}"; do
      for SHOT in "${SHOTS[@]}"; do
        python "$file" \
            --model_name "$MODEL" \
            --dataset_name "$DATA" \
            --samples "$SAMPLE" \
            --shots "$SHOT" \
            --infer_strategy "$infer_strategy" \
            --LRS_threshold "$LRS_TH"
      done
    done
  done
done

######################################
#    Full Inference 实验
######################################

file="exp_1_full.py"
infer_strategy="Full_Inf"   
echo "MODEL=$MODEL DATA=$DATA SAMPLE=$SAMPLE SHOT=$SHOT"

for MODEL in "${MODELS[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    for SAMPLE in "${SAMPLES[@]}"; do
      for SHOT in "${SHOTS[@]}"; do
        python "$file" \
            --model_name "$MODEL" \
            --dataset_name "$DATA" \
            --samples "$SAMPLE" \
            --shots "$SHOT" \
            --infer_strategy "$infer_strategy"

      done
    done
  done
done