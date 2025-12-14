#!/bin/bash

##
##   llama2-7b skip args test
##

#    切换到项目路径
# cd 

MODELS=("llama2-7b")
DATASETS=("cnn_dm_summarization")
SAMPLES=(1000)
SHOTS=(1)
end_layers=(29 27 28)
min_start_layers=(21 22 23)

######################################
#    Dynamic Skip Inference 实验
######################################
file="skip_args_test.py"
echo "Running Dynamic Skip Inference in ${file}"

infer_strategy="Dynamic_Skip_Inf"   

echo "Running Skip Args Test: ${file}"

for MODEL in "${MODELS[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    for SAMPLE in "${SAMPLES[@]}"; do
      for SHOT in "${SHOTS[@]}"; do

        echo "[Skip args test] MODEL=$MODEL DATA=$DATA SAMPLE=$SAMPLE SHOT=$SHOT GPU=$GPU"

        for END in "${end_layers[@]}"; do
          for MIN in "${min_start_layers[@]}"; do

            # echo "end_layer=$END | start_layer=$END | min_start_layer=$MIN"

            python "$file" \
              --cuda 0 \
              --model_name "$MODEL" \
              --dataset_name "$DATA" \
              --samples "$SAMPLE" \
              --shots "$SHOT" \
              --infer_strategy "$infer_strategy" \
              --start_layer "$END" \
              --end_layer "$END" \
              --min_start_layer "$MIN"

          done
        done
      done
    done
  done
done
