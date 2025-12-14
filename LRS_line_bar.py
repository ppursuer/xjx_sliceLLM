import argparse
import torch
import numpy as np
import time

from torchmetrics.text import ROUGEScore
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from get_data import get_data
from model_and_tokenizer import get_tokenizer_and_model

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

""""
cos 和 ΔL2 
图一：折线图
图二：柱状图
"""

class Metric():
    def __init__(self, layers):
        self.LRS_dict = {}
        ## 初始化统计字典
        for i in range(layers):
            self.LRS_dict[f"{i}"] = 0

    def draw_LRS_bar(self, title="LRS Distribution per Layer", save_path=None):
        """
        绘制LRS的柱状图
        """
        # 提取层名和对应的LRS值
        layers = list(self.LRS_dict.keys())
        lrs_values = list(self.LRS_dict.values())
        
        # 创建柱状图
        plt.figure(figsize=(12, 6))
        
        # 绘制柱状图
        bars = plt.bar(layers, lrs_values, color='skyblue', alpha=0.7)
        
        # 添加数值标签
        for bar, value in zip(bars, lrs_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value}', ha='center', va='bottom', fontsize=8)
        
        # 设置图表属性
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Layers')
        plt.ylabel('LRS Values')
        plt.title(title)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LRS bar chart saved to {save_path}")
        
        plt.show()

class LayerwiseHiddenDiffCollector:
    def __init__(self):
        self.records = []  # 每层一条数据：layer_idx, l2_diff, cos_sim
        self.LRS = []
        # self.LRS_dict = {}
    
    def add(self, mode, layer_idx, h_in, h_out, token_str):
        if torch.equal(h_in, h_out):
            self.records.append(self.records[-1].copy())
            return 
        
        h_in_last = h_in
        h_out_last = h_out

        l2 = torch.norm(h_out_last - h_in_last, p=2).item()
        cos = F.cosine_similarity(h_in_last, h_out_last, dim=-1).mean().item()

        # 相对幅度 Relative Magnitude
        h_in_norm = torch.norm(h_in_last, p=2).item()
        rel_mag = l2 / (h_in_norm)        

        self.records.append({
            "mode": mode,
            "layer": layer_idx,
            "token": token_str,
            "l2": l2,
            "cos": cos,
            "rel_mag": rel_mag
        })

    def draw_cos_relmag(self, title="Cosine Difference & Relative Magnitude per Layer", save_path=None):

        # 取出数值
        layers = [rec["layer"] for rec in self.records]
        cos_list = np.array([rec["cos"] for rec in self.records])
        relmag_list = np.array([rec["rel_mag"] for rec in self.records])

        # Cosine difference
        cos_diff = 1 - cos_list

        # ------------------ 绘图 ------------------
        plt.figure(figsize=(10, 5))

        plt.plot(layers, cos_diff, marker="o", linewidth=2, label="1 - Cosine")
        plt.plot(layers, relmag_list, marker="s", linewidth=2, label="Relative Magnitude")

        # 数值标注
        for i in range(len(layers)):
            plt.text(layers[i], cos_diff[i], f"{cos_diff[i]:.2f}",
                     fontsize=7, ha="center", va="bottom")
            plt.text(layers[i], relmag_list[i], f"{relmag_list[i]:.2f}",
                     fontsize=7, ha="center", va="bottom")

        plt.title(title, fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=200)
            print(f"[Saved] Plot saved at: {save_path}")

        plt.show()

    def draw_cos_weight_relmag(self, title="Cosine Difference & Relative Magnitude weighted per Layer", save_path=None):
        # 取出数值
        layers = [rec["layer"] for rec in self.records]
        cos_list = np.array([rec["cos"] for rec in self.records])
        relmag_list = np.array([rec["rel_mag"] for rec in self.records])

        # Cosine difference
        cos_diff = 1 - cos_list

        index = 0.5* cos_diff + 0.5 * relmag_list
        # index = cos_diff * relmag_list
        
        # ------------------ 绘图 ------------------
        plt.figure(figsize=(12, 6))
        
        # 只画index的图
        plt.plot(layers, index, marker='o', linewidth=2, markersize=6, color='red')
        
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Layer')
        plt.ylabel('Combined Index Value')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def compute_LRS(self):
        cos_list = np.array([rec["cos"] for rec in self.records])
        relmag_list = np.array([rec["rel_mag"] for rec in self.records])
        cos_diff = 1 - cos_list
        self.LRS = (0.5* cos_diff + 0.5 * relmag_list).tolist()

    def clustering_by_LRS(self, metric, line=0.2):
        """
        以LRS=0.2划分 浅层 跳跃层 深层
        """
        for idx,iter in enumerate(self.LRS):
            if iter <= line:
                metric.LRS_dict[f"{idx}"] += 1

def forward_full(model, input_ids, tokenizer, collector_full):
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
    hidden_states = model.model.embed_tokens(input_ids)

    for i, layer in enumerate(model.model.layers):
        h_in = hidden_states[:, -1, :].detach()

        hidden_states = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False
        )[0]

        h_out = hidden_states[:, -1, :].detach()

        # 当前层 top1 token
        logits = model.lm_head(model.model.norm(h_out))
        top1_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([top1_id]).strip()
        
        collector_full.add("full", i, h_in, h_out, token_str)

    final_logits = model.lm_head(model.model.norm(hidden_states[:, -1, :]))
    return final_logits

def main(args):
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    evaluation_dataset = get_data(
        args.dataset_name,
        samples=args.samples,
        shots=args.shots,
    )

    tokenizer, model, device = get_tokenizer_and_model(args.model_name, args.cuda)

    metric = Metric(len(model.model.layers))

    for i, data_point in enumerate(tqdm(evaluation_dataset, desc="Evaluating")):
        input_text = data_point.input
        ref_text = data_point.ref

        input_ids_full = tokenizer.encode(input_text, return_tensors="pt").to(device)
        input_len = input_ids_full.shape[1]

        for token_iter in range(args.generate_token):
            collector_full = LayerwiseHiddenDiffCollector()  

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

            # layers, l2_list, cos_list, cos_diff_list, l2_norm_list, score_list = collector_full.cos_l2(token_iter+1)
            collector_full.compute_LRS()
            collector_full.clustering_by_LRS(metric, line=0.2)
            # collector_full.draw_cos_relmag(save_path=f"/home/xiongjunxin/xjx_workplace/xjx_idea/obs_exp/pic/obs16_{token_iter+1}.png")
            # collector_full.draw_cos_weight_relmag(save_path=f"/data/xjx/files/xjx_experiment/pic/LRS/{args.model_name}_{token_iter+1}.png")
        
    metric.draw_LRS_bar(save_path=f"/data/xjx/files/xjx_experiment/pic/LRS/LRS_bar_{args.samples*args.generate_token}.png")
    # print(args.samples*args.generate_token)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index (e.g. 0,1,2,3)")
    parser.add_argument("--model_name", type=str, default="llama2-7b", help="Path to the model")
    parser.add_argument("--dataset_name", type=str, default="xsum_summarization", help="Dataset to evaluate")
    parser.add_argument("--samples", type=int, default=10, help="sample to evaluate")
    parser.add_argument("--shots", type=int, default=1, help="n-shot")
    parser.add_argument("--infer_strategy", type=str, default="Full inf", help="full_inf and mine_inf Static_skip")
    parser.add_argument("--generate_token", type=int, default=60, help="生成token数量")
    args = parser.parse_args()
    main(args)