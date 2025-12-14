import argparse
import torch
import time
import os
import numpy as np

from torchmetrics.text import ROUGEScore
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from get_data import get_data
from model_and_tokenizer import get_tokenizer_and_model

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

"""
motivation 1.1：
层间隐藏状态 的 l2 和 cos
逐token 逐层计算
绘制图
Figure x: Generate one token from Llama2-7B with full inference（blue lines） and skip 26-29 layers（red lines）.
core：除了跳跃层，所有层的输出均相同。
"""

class Full_ROUGEMetric():
    def __init__(self, rouge_keys=("rouge1", "rouge2", "rougeL"), use_stemmer=True):
        self.rouge_metric = ROUGEScore(rouge_keys=rouge_keys, use_stemmer=use_stemmer)
        self.rouge1_list = []
        self.rouge2_list = []
        self.rougeL_list = []
        self.avgconfid = []
        self.avglayer = []
        self.time_list = []
        self.EE_layer = []

class LayerwiseHiddenDiffCollector:
    def __init__(self):
        self.records = []  # 每层一条数据：layer_idx, l2_diff, cos_sim

    def add(self, mode, layer_idx, h_in, h_out, token_str):
        if torch.equal(h_in, h_out):
            self.records.append(self.records[-1].copy())
            return 
        
        h_in_last = h_in[:, -1, :]
        h_out_last = h_out[:, -1, :]

        l2 = torch.norm(h_out_last - h_in_last, p=2).item()
        cos = F.cosine_similarity(h_in_last, h_out_last, dim=-1).mean().item()

        self.records.append({
            "mode": mode,
            "layer": layer_idx,
            "token": token_str,
            "l2": l2,
            "cos": cos,
        })

def draw_two(full_list, skip_list, idx, title="Full vs Skip Layerwise Hidden Diff"):
    tokens_full = [r["token"] for r in full_list]
    tokens_skip = [r["token"] for r in skip_list]

    l2_full  = [r["l2"] for r in full_list]
    l2_skip  = [r["l2"] for r in skip_list]

    cos_full = [r["cos"] for r in full_list]
    cos_skip = [r["cos"] for r in skip_list]

    # ==== X轴长度 ====
    num_layers = len(full_list)
    x = np.arange(num_layers)

    # 自适应图宽
    fig_width = max(10, num_layers * 0.4)
    fig, ax1 = plt.subplots(figsize=(fig_width, 5))
    ax2 = ax1.twinx()

    # ==== 左轴 L2 ====
    ax1.plot(x, l2_full, color="tab:blue", marker="o", label="Full L2")
    ax1.plot(x, l2_skip, color="tab:red", marker="o", label="Skip L2", linestyle="--")

    ax1.set_ylabel("L2 Difference")

    # ==== 右轴 Cosine ====
    ax2.plot(x, cos_full, color="tab:blue", marker="x", linestyle="-.", label="Full CosSim")
    ax2.plot(x, cos_skip, color="tab:red", marker="x", linestyle=":", label="Skip CosSim")
    ax2.set_ylabel("Cosine Similarity")

    # ==== 下方 X 轴 (Full token) ====
    ax1.set_xticks(x)
    ax1.set_xticklabels(tokens_full, rotation=70, ha="right", fontsize=8)
    ax1.set_xlabel("Full Inference", labelpad=-8)

    # ==== 上方 X 轴（空白 label，不显示 token） ====
    ax_top = ax1.secondary_xaxis('top')
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(tokens_skip, rotation=70, ha="left", fontsize=8)  
    ax_top.set_xlabel("Skip Inference", labelpad=-8)                      # 不显示标题

    # ==== 样式 ====
    plt.title(title, pad=75)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ==== 合并图例 ====
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.tight_layout()

    save_path = f"/root/lniox/slo_mrqs/xjx_experiment/pic/motiv_1_1"
    os.makedirs(save_path, exist_ok=True)
    if save_path:
        plt.savefig(f"{save_path}/{args.model_name}_token_{idx}.png", dpi=300, bbox_inches="tight")
        print(f"图已保存到{save_path}/token_{idx}.png")
    else:
        plt.show()

def forward_full(model, input_ids, tokenizer, collector_full):
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

    hidden_states = model.model.embed_tokens(input_ids)

    for i, layer in enumerate(model.model.layers):
        h_in = hidden_states.clone()

        hidden_states = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False
        )[0]

        h_out = hidden_states.clone()

        # 当前层 top1 token
        logits = model.lm_head(model.model.norm(h_out[:, -1, :]))
        top1_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([top1_id]).strip()

        collector_full.add("full", i, h_in, h_out, token_str)

    final_logits = model.lm_head(model.model.norm(hidden_states[:, -1, :]))
    return final_logits

def forward_skip(model, input_ids, tokenizer, collector_skip, SKIP_START, SKIP_END):
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

    hidden_states = model.model.embed_tokens(input_ids)

    for i, layer in enumerate(model.model.layers):
        h_in = hidden_states.clone()
        # print(h_in)
        if SKIP_START <= i <= SKIP_END:
            # 跳过层：不更新
            h_out = hidden_states.clone()
            # print(h_out)
        else:
            # 正常前向
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                use_cache=False
            )[0]
            h_out = hidden_states.clone()
            # print(h_out)

        # 当前层 top1 token
        logits = model.lm_head(model.model.norm(h_out[:, -1, :]))
        top1_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([top1_id]).strip()

        collector_skip.add("skip", i, h_in, h_out, token_str)

    final_logits = model.lm_head(model.model.norm(hidden_states[:, -1, :]))
    return final_logits

def main(args):
    MAX_NEW_TOKENS = 30
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    evaluation_dataset = get_data(
                                args.dataset_name, 
                                samples=args.samples, 
                                shots=args.shots
                                )

    tokenizer, model, device = get_tokenizer_and_model(args.model_name, args.cuda)

    if len(model.model.layers) == 40:
        SKIP_START = 30
        SKIP_END = 35
    elif len(model.model.layers) == 32:
        SKIP_START = 25
        SKIP_END = 28

    for i, data_point in enumerate(tqdm(evaluation_dataset, desc="Evaluating")):
            
        input_text = data_point.input
        ref_text = data_point.ref

        input_ids_full = tokenizer.encode(input_text, return_tensors="pt").to(device)
        input_ids_skip = tokenizer.encode(input_text, return_tensors="pt").to(device)
        input_len = input_ids_full.shape[1]

        for token_iter in range(MAX_NEW_TOKENS):
            collector_full = LayerwiseHiddenDiffCollector()
            collector_skip = LayerwiseHiddenDiffCollector()

            ################# full FORWARD ################# 
            full_logits = forward_full(model,
                                       input_ids_full, 
                                       tokenizer, 
                                       collector_full
                                       )
            probs = torch.softmax(full_logits, dim=-1)
            max_prob, next_token_id = torch.max(probs, dim=-1)

            confidence = max_prob.item()  
            input_ids_full = torch.cat([input_ids_full, next_token_id.unsqueeze(0)], dim=-1)
            print(f"FULL FORWARD Token {token_iter+1} | Conf: {confidence:.4f} | Token: {tokenizer.decode([next_token_id.item()])}")

            ################# skip FORWARD #################
            skip_logits = forward_skip(model, 
                                       input_ids_skip, 
                                       tokenizer, 
                                       collector_skip, 
                                       SKIP_START, 
                                       SKIP_END
                                       )
            probs = torch.softmax(skip_logits, dim=-1)
            max_prob, next_token_id = torch.max(probs, dim=-1)

            confidence = max_prob.item()  
            input_ids_skip = torch.cat([input_ids_skip, next_token_id.unsqueeze(0)], dim=-1)
            print(f"SKIP FORWARD Token {token_iter+1} | Conf: {confidence:.4f} | Token: {tokenizer.decode([next_token_id.item()])}")

            draw_two(collector_full.records, collector_skip.records, token_iter+1)

        print(f"reference:{ref_text}")
        metric = ROUGEScore()
        generated_text = tokenizer.decode(input_ids_full[0, input_len:], skip_special_tokens=True)
        print(f"Full forward\n第{i+1}条数据模型输出：{generated_text}")
        scores_full = metric(preds=[generated_text], target=[ref_text])

        rouge1_full = scores_full["rouge1_fmeasure"].item()
        rouge2_full = scores_full["rouge2_fmeasure"].item()
        rougeL_full = scores_full["rougeL_fmeasure"].item()

        print(f"[FULL] Rouge-1: {rouge1_full:.4f}")
        print(f"[FULL] Rouge-2: {rouge2_full:.4f}")
        print(f"[FULL] Rouge-L: {rougeL_full:.4f}")


        generated_text = tokenizer.decode(input_ids_skip[0, input_len:], skip_special_tokens=True)
        print(f"Skip forward\n第{i+1}条数据模型输出：{generated_text}")
        scores_skip = metric(preds=[generated_text], target=[ref_text])

        rouge1_skip = scores_skip["rouge1_fmeasure"].item()
        rouge2_skip = scores_skip["rouge2_fmeasure"].item()
        rougeL_skip = scores_skip["rougeL_fmeasure"].item()

        print(f"[SKIP] Rouge-1: {rouge1_skip:.4f}")
        print(f"[SKIP] Rouge-2: {rouge2_skip:.4f}")
        print(f"[SKIP] Rouge-L: {rougeL_skip:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index (e.g. 0,1,2,3)")
    parser.add_argument("--model_name", type=str, default="llama2-7b", help="Path to the model")
    parser.add_argument("--dataset_name", type=str, default="xsum_summarization", help="Dataset to evaluate")
    parser.add_argument("--samples", type=int, default=1, help="sample to evaluate")
    parser.add_argument("--shots", type=int, default=1, help="n-shot")
    parser.add_argument("--infer_strategy", type=str, default="Full inf", help="full_inf and mine_inf Static_skip")
    args = parser.parse_args()
    main(args)

