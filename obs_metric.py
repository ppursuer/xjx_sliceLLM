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
import matplotlib.pyplot as plt

from get_data import get_data
from model_and_tokenizer import get_tokenizer_and_model
from metric_collector import Metric_Full, LayerwiseHiddenDiffCollector_full
from inference import forward_full

"""
obs metrics with 图 
"""

def draw_token_metrics(records, metric_key_1, metric_key_2, token_iter):
    
    tokens = [r["token"] for r in records]

    if metric_key_1 == "1-cos":
        v1 = [1.0 - r["cos"] for r in records]
    else :
        v1 = [r[metric_key_1] for r in records]

    if metric_key_2 == "1-cos":
        v2 = [1.0 - r["cos"] for r in records]
    else :
        v2 = [r[metric_key_2] for r in records]

    x = np.arange(len(tokens))

    fig, ax1 = plt.subplots(figsize=(max(10, len(tokens) * 0.4), 5))
    ax2 = ax1.twinx()

    ax1.plot(x, v1, marker="o", color="tab:blue", label=metric_key_1)
    ax2.plot(x, v2, marker="x", color="tab:red", linestyle="--", label=metric_key_2)

    ax1.set_xlabel("Generated Token")
    ax1.set_ylabel(metric_key_1)
    ax2.set_ylabel(metric_key_2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(tokens, rotation=70, ha="right", fontsize=8)

    plt.title(f"{metric_key_1} & {metric_key_2} vs Token")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_path = f"./pic/obs_metrics/{token_iter}_{metric_key_1}_{metric_key_2}.png"
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()


def draw_token_metric(records, metric_key, token_iter):
    
    tokens = [r["token"] for r in records]

    if metric_key == "cos+rel_mag":
        values = [
                (r["cos"]) + r["rel_mag"]
                for r in records
            ]
    else:
        if metric_key == "1-cos":
            values = [1.0 - r["cos"] for r in records]
        else :
            values = [r[metric_key] for r in records]

    x = np.arange(len(tokens))

    plt.figure(figsize=(max(10, len(tokens) * 0.4), 5))
    plt.plot(x, values, marker="o", linewidth=2)

    plt.xticks(x, tokens, rotation=70, ha="right", fontsize=8)
    plt.xlabel("Generated Token")
    plt.ylabel(metric_key)
    plt.title(f"Token-wise {metric_key}")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_path = f"./pic/obs_metrics/{token_iter}_{metric_key}.png"
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()

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
                # start_time = time.time()
                next_token_id, next_token_str, input_ids, infer_time = forward_full(model,
                                                                    input_ids, 
                                                                    tokenizer, 
                                                                    collector_full  
                                                                    )
                token_time += infer_time

                # draw_token_metric(collector_full.records, "1-cos", token_iter+1)
                # draw_token_metric(collector_full.records, "l2", token_iter+1)
                # draw_token_metric(collector_full.records, "rel_mag", token_iter+1)

                draw_token_metric(collector_full.records, "cos+rel_mag", token_iter+1)

                # draw_token_metrics(collector_full.records, "1-cos", "rel_mag", token_iter+1)

            generated_text = tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True)
            r1, r2, rl = metric_full.add(generated_text, ref_text, token_time, token_iter+1)
            torch.cuda.empty_cache() 
            print(f"第{i+1}条数据模型输出：{generated_text}")
            print("rouge-1:", r1)
            print("rouge-2:", r2)
            print("rouge-L:", rl)
            print("token num:", token_iter+1)
            # print(f"inference time: {token_time:.4f}")
            # print(collector_full.records)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index (e.g. 0,1,2,3)")
    parser.add_argument("--model_name", type=str, default="llama2-7b", help="Path to the model")
    parser.add_argument("--dataset_name", type=str, default="cnn_dm_summarization", help="Dataset to evaluate")
    parser.add_argument("--samples", type=int, default=1, help="sample to evaluate")
    parser.add_argument("--shots", type=int, default=1, help="n-shot")
    parser.add_argument("--infer_strategy", type=str, default="Full_Inf", help="full_inf and skip_inf")
    parser.add_argument("--max_generate_token", type=int, default=10, help="最大生成token数量")
    args = parser.parse_args()
    main(args)