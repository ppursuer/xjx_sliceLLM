import argparse
import torch
import time
import os
import numpy as np

from torchmetrics.text import ROUGEScore
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt

from get_data import get_data
from model_and_tokenizer import get_tokenizer_and_model, pre_setup_skip

from metric_collector import Metric_Skip, LayerwiseHiddenDiffCollector_skip
from inference import forward_skip, forward_skip_update

"""
skip 
llama2-7b 限制 （22层-29层】   上限是7层

1000条数据
第一组 29 29 22


llama2-13b ：限制（25层-37层】 上限是12层
"""

def main(args):
    exp_start_time = time.time()
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    evaluation_dataset = get_data(
        args.dataset_name,
        samples=args.samples,
        shots=args.shots,
    )

    tokenizer, model, device = get_tokenizer_and_model(args.model_name, args.cuda)

    metric_skip = Metric_Skip()

    stop_tokens = [
        ".", "!", "?",          # 句子终止符
        "\n", "\t",             # 换行 / tab
    ]

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))   
    # 在当前路径平级的 logs 目录
    log_dir = os.path.join(current_dir, f"logs/args_test/{args.model_name}/{args.dataset_name}")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 构建日志文件路径
    log_path = os.path.join(log_dir, f"{len(evaluation_dataset)}_{timestamp}.log")

    for i, data_point in enumerate(tqdm(evaluation_dataset, desc="Evaluating")):
        input_text = data_point.input
        ref_text = data_point.ref

        start_layer = args.start_layer
        end_layer = args.end_layer

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        input_len = input_ids.shape[1] 

        total_skip_layer = 0
        total_generate_token = 0
        token_time = 0
        # start_time = time.time() 
        # Use inference mode for the token generation loop
        with torch.inference_mode():        
            for token_iter in range(args.max_generate_token):
                collector_skip = LayerwiseHiddenDiffCollector_skip()  

                ################# SKIP FORWARD ################# 
                # start_time = time.time()
                next_token_id, next_token_str, input_ids, start_layer, skip_sum, infer_time = forward_skip_update(model,
                                                            input_ids, 
                                                            tokenizer, 
                                                            collector_skip,
                                                            start_layer,
                                                            end_layer,
                                                            args.min_start_layer,
                                                            args.LRS_threshold,
                                                            )
                token_time += infer_time
                # end_time = time.time()
                # infer_time += end_time - start_time
                # print(skip_sum)
                total_skip_layer += skip_sum
                # if args.dataset_name == "xsum_summarization" and next_token_id == tokenizer.eos_token_id or any(s in next_token_str for s in stop_tokens):
                #     break

            # end_time = time.time()
            # infer_time = end_time-start_time
            if total_skip_layer ==0 or token_iter ==0:
                avg_skip_layer = 0
            else:
                avg_skip_layer = total_skip_layer / (token_iter+1)
            generated_text = tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True)
            r1, r2, rl = metric_skip.add(generated_text, ref_text, avg_skip_layer, token_time, token_iter+1)
            torch.cuda.empty_cache()
            # print(f"第{i+1}条数据模型输出：{generated_text}")
            # print("rouge-1:", r1)
            # print("rouge-2:", r2)
            # print("rouge-L:", rl)
            # print("token num:", token_iter+1)
            # print("avg skip:", avg_skip_layer)
            # print(f"inference time: {token_time:.4f}")       

        ## 每1000条数据保存一次日志
        if (i+1) % 1000 == 0:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"=====前{i+1}条数据=====\n")
                f.write(metric_skip.summary_formatted())
                f.write("\n")

    exp_end_time = time.time()
    exp_total_time = exp_end_time - exp_start_time
    # 转换为 HH:MM:SS 格式
    hours = int(exp_total_time  // 3600)
    minutes = int((exp_total_time  % 3600) // 60)
    seconds = int(exp_total_time  % 60)
    formatted_time_2 = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    # # 获取当前脚本所在目录
    # current_dir = os.path.dirname(os.path.abspath(__file__))   
    # # 在当前路径平级的 logs 目录
    # log_dir = os.path.join(current_dir, f"logs/args_test/{args.model_name}/{args.dataset_name}")
    # os.makedirs(log_dir, exist_ok=True)

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # # 构建日志文件路径
    # log_path = os.path.join(log_dir, f"{len(evaluation_dataset)}_{timestamp}.log")
    
    # 生成要写入的文本内容
    summary_text = (
        "\n================= Overall Evaluation =================\n"
        f"Model: {args.model_name}\n"
        f"Dataset: {args.dataset_name}\n"
        f"Sample: {len(evaluation_dataset)}\n"
        f"Max Generate len: {args.max_generate_token}\n"
        f"Few shot: {args.shots}\n"
        f"Inference strategy: {args.infer_strategy}\n"
        f"Experiment Total Time: {formatted_time_2}\n"
        + metric_skip.summary_formatted() + 
        f"End layer: {args.end_layer}\n"
        f"Start layer: {args.start_layer}\n"
        f"Min start layer: {args.min_start_layer}\n"
        "=======================================================\n"
    )
    print(summary_text)
    # # 写入日志文件（追加模式）
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Evaluation results saved to: {log_path}")

    # # 保存整个数据集上的skip inference输出结果
    # csv_path = os.path.join(log_dir, f"output.csv")
    # metric_skip.save_output_csv(csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index (e.g. 0,1,2,3)")
    parser.add_argument("--model_name", type=str, default="llama2-7b", help="Path to the model")
    parser.add_argument("--dataset_name", type=str, default="xsum_summarization", help="Dataset to evaluate")
    parser.add_argument("--samples", type=int, default=10, help="sample to evaluate")
    parser.add_argument("--shots", type=int, default=1, help="n-shot")
    parser.add_argument("--infer_strategy", type=str, default="Dynamic_Skip_Inf", help="full_inf and skip_inf")
    parser.add_argument("--max_generate_token", type=int, default=60, help="生成token数量")
    
    parser.add_argument("--end_layer", type=int, default=29, help="跳层结束层")
    parser.add_argument("--start_layer", type=int, default=29, help="跳层开始层")
    parser.add_argument("--min_start_layer", type=int, default=22, help="")
    parser.add_argument("--LRS_threshold", type=float, default=0.175, help="阈值")

    args = parser.parse_args()
    # args.end_layer,args.start_layer,args.min_start_layer = pre_setup_skip(args.model_name)
    # args.end_layer,args.start_layer,args.min_start_layer = 
    main(args)