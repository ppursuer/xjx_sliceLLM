#!/bin/bash

##
##   llama2-7b run full
##

#    切换到项目路径
# cd 

###############################
#         GPU 轮询逻辑
###############################
# THRESHOLD=20000        # 20GB
# CHECK_INTERVAL=300      # 每 300s 检查一次

# echo "Waiting for a GPU with >= ${THRESHOLD} MB free memory..."


# CHOSEN_GPU=-1

# while true; do
#     GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

#     for ((i=0; i<GPU_COUNT; i++)); do
#         FREE=$(nvidia-smi \
#             --query-gpu=memory.free \
#             --format=csv,noheader,nounits | sed -n "$((i+1))p")

#         echo "GPU $i Free: ${FREE} MB"

#         if [ "$FREE" -ge "$THRESHOLD" ]; then
#             CHOSEN_GPU=$i
#             break
#         fi
#     done

#     if [ "$CHOSEN_GPU" -ge 0 ]; then
#         echo "Selected GPU: $CHOSEN_GPU (Free ${FREE} MB)"
#         export CUDA_VISIBLE_DEVICES=$CHOSEN_GPU
#         break
#     fi

#     # echo "No suitable GPU yet, retrying in ${CHECK_INTERVAL}s..."
#     sleep ${CHECK_INTERVAL}
# done

MODELS=("llama2-7b")
DATASETS=("cnn_dm_summarization")
SAMPLES=(1000)
SHOTS=(1)

# python exp_1_skip_update.py \
#         --cuda 0 \
#         --model_name llama2-7b \
#         --dataset_name cnn_dm_summarization \

######################################
#    Full Inference 实验
######################################
file="exp_1_full.py"
echo "Running Dynamic Skip Inference in ${file}"

infer_strategy="Full_Inf"   

echo "Running Skip Args Test: ${file}"

for MODEL in "${MODELS[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    for SAMPLE in "${SAMPLES[@]}"; do
      for SHOT in "${SHOTS[@]}"; do

        echo "MODEL=$MODEL DATA=$DATA SAMPLE=$SAMPLE SHOT=$SHOT GPU=$GPU"
        python "$file" \
            --model_name "$MODEL" \
            --dataset_name "$DATA" \
            --samples "$SAMPLE" \
            --shots "$SHOT" \
            --infer_strategy "$infer_strategy" \
          done
        done
      done
    done
  done


