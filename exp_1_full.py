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
from model_and_tokenizer import get_tokenizer_and_model
from metric_collector import Metric_Full, LayerwiseHiddenDiffCollector_full
from inference import forward_full

"""
Full inference  log output
"""

def main(args):
    exp_start_time = time.time()
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    evaluation_dataset = get_data(
        args.dataset_name,
        samples = args.samples,
        shots=args.shots,
    )

    tokenizer, model, device = get_tokenizer_and_model(args.model_name, args.cuda)

    metric_full = Metric_Full()

    stop_tokens = [
        ".", "!", "?",          # 句子终止符
        "\n", "\t",             # 换行 / tab
    ]

    for i, data_point in enumerate(tqdm(evaluation_dataset, desc="Evaluating")):
        input_text = data_point.input
        ref_text = data_point.ref

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        input_len = input_ids.shape[1]

        # start_time = time.time()
        token_time = 0
        with torch.inference_mode():
            for token_iter in range(args.max_generate_token):
                collector_full = LayerwiseHiddenDiffCollector_full()

                ################# FULL FORWARD #################
                # start_time = time.time()
                next_token_id, next_token_str, input_ids, infer_time = forward_full(model,
                                                                    input_ids, 
                                                                    tokenizer, 
                                                                    collector_full  
                                                                    )
                token_time += infer_time
                # end_time = time.time()
                # Case 1: EOS token Case 2: 文本匹配（句号/换行等）
                # if args.dataset_name == "xsum_summarizaiton" and next_token_id == tokenizer.eos_token_id or any(s in next_token_str for s in stop_tokens):
                #     break

            # end_time = time.time()
            # infer_time = end_time-start_time
            generated_text = tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True)
            r1, r2, rl = metric_full.add(generated_text, ref_text, token_time, token_iter+1)
            torch.cuda.empty_cache() 
            # print(f"第{i+1}条数据模型输出：{generated_text}")
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
    log_dir = os.path.join(current_dir, f"logs/{args.infer_strategy}/{args.model_name}/{args.dataset_name}")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 构建日志文件路径 log
    log_path = os.path.join(log_dir, f"{len(evaluation_dataset)}_{args.max_generate_token}_{timestamp}.log")
    
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

    # 保存整个数据集上的full inference输出结果
    # csv_path = os.path.join(log_dir, f"output.csv")
    # metric_full.save_output_csv(csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index (e.g. 0,1,2,3)")
    parser.add_argument("--model_name", type=str, default="llama2-7b", help="Path to the model")
    parser.add_argument("--dataset_name", type=str, default="cnn_dm_summarization", help="Dataset to evaluate")
    parser.add_argument("--samples", type=int, default=1000, help="sample to evaluate")
    parser.add_argument("--shots", type=int, default=1, help="n-shot")
    parser.add_argument("--infer_strategy", type=str, default="Full_Inf", help="full_inf and skip_inf")
    parser.add_argument("--max_generate_token", type=int, default=60, help="最大生成token数量")
    args = parser.parse_args()
    main(args)