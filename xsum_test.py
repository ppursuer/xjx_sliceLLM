import argparse
import torch
import time
import os
import numpy as np
import time
import torch

from torchmetrics.text import ROUGEScore
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F

from get_data import get_data
from model_and_tokenizer import get_tokenizer_and_model, pre_setup_skip
from metric_collector import Metric_Full, Metric_Skip, LayerwiseHiddenDiffCollector_full, LayerwiseHiddenDiffCollector_skip
from inference import forward_full, forward_skip_update

"""
test xsum 
max token or one sentence
"""

def run_full_one(args, evaluation_dataset, tokenizer, model, device):
    exp_start_time = time.time()

    metric_full = Metric_Full()

    stop_tokens = [
        ".", "!", "?",          # 句子终止符
        "\n", "\t",             # 换行 / tab
    ]

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 在当前路径平级的 logs 目录
    log_dir = os.path.join(current_dir, f"logs/test_xsum")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 构建日志文件路径 log
    log_path = os.path.join(log_dir, f"full_one_{len(evaluation_dataset)}_{timestamp}.log")

    for i, data_point in enumerate(tqdm(evaluation_dataset, desc="Evaluating")):
        input_text = data_point.input
        ref_text = data_point.ref

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        input_len = input_ids.shape[1]

        token_time = 0
        with torch.inference_mode():
            for token_iter in range(args.max_generate_token):
                collector_full = LayerwiseHiddenDiffCollector_full()

                ################# FULL FORWARD #################
                next_token_id, next_token_str, input_ids, infer_time = forward_full(model,
                                                                    input_ids, 
                                                                    tokenizer, 
                                                                    collector_full  
                                                                    )
                token_time += infer_time
                if next_token_id == tokenizer.eos_token_id or any(s in next_token_str for s in stop_tokens):
                    break

            generated_text = tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True)
            r1, r2, rl = metric_full.add(generated_text, ref_text, token_time, token_iter+1)
            torch.cuda.empty_cache() 
            # print(f"第{i+1}条数据模型输出：{generated_text}")
            # print(f"第{i+1}条数据: {ref_text}")
            # print("rouge-1:", r1)
            # print("rouge-2:", r2)
            # print("rouge-L:", rl)
            # print("token num:", token_iter+1)
            # print(f"inference time: {token_time:.4f}")

        ## 每1000条数据保存一次日志
        if (i+1) % 1000 == 0:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"=====前{i+1}条数据=====\n")
                f.write(metric_full.summary_formatted())
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
    # log_dir = os.path.join(current_dir, f"logs/test_xsum")
    # os.makedirs(log_dir, exist_ok=True)

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # # 构建日志文件路径 log
    # log_path = os.path.join(log_dir, f"full_one_{len(evaluation_dataset)}_{timestamp}.log")
    
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
        + metric_full.summary_formatted() + 
        "=======================================================\n"
    )
    print(summary_text)
    # 写入日志文件（追加模式）
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Evaluation results saved to: {log_path}")

def run_full_max(args, evaluation_dataset, tokenizer, model, device):
    exp_start_time = time.time()

    metric_full = Metric_Full()

    # stop_tokens = [
    #     ".", "!", "?",          # 句子终止符
    #     "\n", "\t",             # 换行 / tab
    # ]

    for i, data_point in enumerate(tqdm(evaluation_dataset, desc="Evaluating")):
        input_text = data_point.input
        ref_text = data_point.ref

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        input_len = input_ids.shape[1]

        token_time = 0
        with torch.inference_mode():
            for token_iter in range(args.max_generate_token):
                collector_full = LayerwiseHiddenDiffCollector_full()

                ################# FULL FORWARD #################
                next_token_id, next_token_str, input_ids, infer_time = forward_full(model,
                                                                    input_ids, 
                                                                    tokenizer, 
                                                                    collector_full  
                                                                    )
                token_time += infer_time
                # if next_token_id == tokenizer.eos_token_id or any(s in next_token_str for s in stop_tokens):
                #     break

            generated_text = tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True)
            r1, r2, rl = metric_full.add(generated_text, ref_text, token_time, token_iter+1)
            torch.cuda.empty_cache() 
            # print(f"第{i+1}条数据模型输出：{generated_text}")
            # print(f"第{i+1}条数据: {ref_text}")
            # print("rouge-1:", r1)
            # print("rouge-2:", r2)
            # print("rouge-L:", rl)
            # print("token num:", token_iter+1)
            # print(f"inference time: {token_time:.4f}")

    exp_end_time = time.time()
    exp_total_time = exp_end_time - exp_start_time
    # 转换为 HH:MM:SS 格式
    hours = int(exp_total_time  // 3600)
    minutes = int((exp_total_time  % 3600) // 60)
    seconds = int(exp_total_time  % 60)
    formatted_time_2 = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 在当前路径平级的 logs 目录
    log_dir = os.path.join(current_dir, f"logs/test_xsum")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 构建日志文件路径 log
    log_path = os.path.join(log_dir, f"full_max_{timestamp}.log")
    
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
        + metric_full.summary_formatted() + 
        "=======================================================\n"
    )
    print(summary_text)
    # 写入日志文件（追加模式）
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Evaluation results saved to: {log_path}")

def run_skip_one(args, evaluation_dataset, tokenizer, model, device):
    exp_start_time = time.time()
    metric_skip = Metric_Skip()

    stop_tokens = [
        ".", "!", "?",          # 句子终止符
        "\n", "\t",             # 换行 / tab
    ]

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))   
    # 在当前路径平级的 logs 目录
    log_dir = os.path.join(current_dir, f"logs/test_xsum")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 构建日志文件路径
    log_path = os.path.join(log_dir, f"skip_one_{len(evaluation_dataset)}_{timestamp}.log")


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
        # Use inference mode for the token generation loop
        with torch.inference_mode():        
            for token_iter in range(args.max_generate_token):
                collector_skip = LayerwiseHiddenDiffCollector_skip()  

                ################# SKIP FORWARD ################# 
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
                # print(skip_sum)
                total_skip_layer += skip_sum
                if next_token_id == tokenizer.eos_token_id or any(s in next_token_str for s in stop_tokens):
                    break

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
            # print(f"第{i+1}条数据: {ref_text}")
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
        f"LRS threshold: {args.LRS_threshold}\n"
        "=======================================================\n"
    )
    print(summary_text)
    # 写入日志文件（追加模式）
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Evaluation results saved to: {log_path}")


def run_skip_max(args, evaluation_dataset, tokenizer, model, device):
    exp_start_time = time.time()
    metric_skip = Metric_Skip()

    # stop_tokens = [
    #     ".", "!", "?",          # 句子终止符
    #     "\n", "\t",             # 换行 / tab
    # ]

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
        # Use inference mode for the token generation loop
        with torch.inference_mode():        
            for token_iter in range(args.max_generate_token):
                collector_skip = LayerwiseHiddenDiffCollector_skip()  

                ################# SKIP FORWARD ################# 
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
                # print(skip_sum)
                total_skip_layer += skip_sum
                # if next_token_id == tokenizer.eos_token_id or any(s in next_token_str for s in stop_tokens):
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
            # print(f"第{i+1}条数据: {ref_text}")
            # print("rouge-1:", r1)
            # print("rouge-2:", r2)
            # print("rouge-L:", rl)
            # print("token num:", token_iter+1)
            # print("avg skip:", avg_skip_layer)
            # print(f"inference time: {token_time:.4f}")       

    exp_end_time = time.time()
    exp_total_time = exp_end_time - exp_start_time
    # 转换为 HH:MM:SS 格式
    hours = int(exp_total_time  // 3600)
    minutes = int((exp_total_time  % 3600) // 60)
    seconds = int(exp_total_time  % 60)
    formatted_time_2 = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))   
    # 在当前路径平级的 logs 目录
    log_dir = os.path.join(current_dir, f"logs/test_xsum")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 构建日志文件路径
    log_path = os.path.join(log_dir, f"skip_max_{timestamp}.log")
    
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
        f"LRS threshold: {args.LRS_threshold}\n"
        "=======================================================\n"
    )
    print(summary_text)
    # 写入日志文件（追加模式）
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Evaluation results saved to: {log_path}")

def main(args):
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    evaluation_dataset = get_data(
        args.dataset_name,
        samples = args.samples,
        shots=args.shots,
    )

    tokenizer, model, device = get_tokenizer_and_model(args.model_name, args.cuda)

    run_full_one(args, evaluation_dataset, tokenizer, model, device)
    # run_full_max(args, evaluation_dataset, tokenizer, model, device)

    run_skip_one(args, evaluation_dataset, tokenizer, model, device)
    # run_skip_max(args, evaluation_dataset, tokenizer, model, device)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index (e.g. 0,1,2,3)")
    parser.add_argument("--model_name", type=str, default="llama2-7b", help="Path to the model")
    parser.add_argument("--dataset_name", type=str, default="xsum_summarization", help="Dataset to evaluate")
    parser.add_argument("--samples", type=int, default=-1, help="sample to evaluate")
    parser.add_argument("--shots", type=int, default=1, help="n-shot")
    parser.add_argument("--infer_strategy", type=str, default="Full_Inf", help="full_inf and skip_inf")
    parser.add_argument("--max_generate_token", type=int, default=60, help="最大生成token数量")
    parser.add_argument("--LRS_threshold", type=float, default=0.185, help="阈值")
    args = parser.parse_args()
    args.end_layer,args.start_layer,args.min_start_layer = pre_setup_skip(args.model_name)
    main(args)