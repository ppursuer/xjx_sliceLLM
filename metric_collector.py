from torchmetrics.text import ROUGEScore
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

class Metric_Full:
    def __init__(self, rouge_keys=("rouge1", "rouge2", "rougeL"), use_stemmer=True):
        self.rouge_metric = ROUGEScore(rouge_keys=rouge_keys, use_stemmer=use_stemmer)
        self.rouge1_list = []
        self.rouge2_list = []
        self.rougeL_list = []
        self.infer_time_list = []
        self.output = []
        self.ref = []
        self.token_generate_sum = []

    def add(self, pred, ref, infer_time, token_sum):
        """添加一条样本的 ROUGE"""
        r1, r2, rL = self.compute_single(pred, ref)
        self.rouge1_list.append(r1)
        self.rouge2_list.append(r2)
        self.rougeL_list.append(rL)
        self.ref.append(ref)
        self.output.append(pred)
        self.infer_time_list.append(infer_time)
        self.token_generate_sum.append(token_sum)
        return r1, r2, rL 

    def compute_single(self, pred, ref):
        scores = self.rouge_metric(preds=[pred], target=[ref])
        r1 = scores["rouge1_fmeasure"].item()
        r2 = scores["rouge2_fmeasure"].item()
        rL = scores["rougeL_fmeasure"].item()
        return r1, r2, rL
    
    def compute_avg(self):
        return {
            "rouge1": np.mean(self.rouge1_list),
            "rouge2": np.mean(self.rouge2_list),
            "rougeL": np.mean(self.rougeL_list),
            "token_generate_sum": np.mean(self.token_generate_sum),
            "time": np.mean(self.infer_time_list),
        }

    def summary_formatted(self):
        avg = self.compute_avg()
        return (
            f"Average Rouge-1 (F): {avg['rouge1']:.4f}\n"
            f"Average Rouge-2 (F): {avg['rouge2']:.4f}\n"
            f"Average Rouge-L (F): {avg['rougeL']:.4f}\n"
            f"Average Token Generate Sum: {avg['token_generate_sum']:.4f}\n"
            f"Average Infer Time: {avg['time']:.4f}\n"
        )
    
    def save_output_csv(self, path):
        """按顺序保存每条样本的预测与参考"""
        df = pd.DataFrame({
            "prediction": self.output,
            "reference": self.ref,
            "rouge1": self.rouge1_list,
            "rouge2": self.rouge2_list,
            "rougeL": self.rougeL_list,
        })

        df.to_csv(path, index=False, encoding="utf-8")
        print(f"[Saved] Output saved to {path}, total {len(df)} samples.")

class Metric_Skip:
    def __init__(self, rouge_keys=("rouge1", "rouge2", "rougeL"), use_stemmer=True):
        self.rouge_metric = ROUGEScore(rouge_keys=rouge_keys, use_stemmer=use_stemmer)
        self.rouge1_list = []
        self.rouge2_list = []
        self.rougeL_list = []
        self.skip_layer_list = []
        self.infer_time_list = []
        self.output = []
        self.ref = []
        self.token_generate_sum = []
        self.LRS_dict = {}

    def add(self, pred, ref, avg_skip_layer, infer_time, token_sum):
        """添加一条样本的 ROUGE"""
        r1, r2, rL = self.compute_single(pred, ref)
        self.rouge1_list.append(r1)
        self.rouge2_list.append(r2)
        self.rougeL_list.append(rL)
        self.skip_layer_list.append(avg_skip_layer)
        self.infer_time_list.append(infer_time)
        self.ref.append(ref)
        self.output.append(pred)
        self.token_generate_sum.append(token_sum)
        return r1, r2, rL
    
    def compute_single(self, pred, ref):
        scores = self.rouge_metric(preds=[pred], target=[ref])
        r1 = scores["rouge1_fmeasure"].item()
        r2 = scores["rouge2_fmeasure"].item()
        rL = scores["rougeL_fmeasure"].item()
        return r1, r2, rL
    
    def compute_avg(self):
        return {
            "rouge1": np.mean(self.rouge1_list),
            "rouge2": np.mean(self.rouge2_list),
            "rougeL": np.mean(self.rougeL_list),
            "token_generate_sum": np.mean(self.token_generate_sum),
            "avg_skip_layer": np.mean(self.skip_layer_list),
            "time": np.mean(self.infer_time_list),
        }
    
    def summary_formatted(self):
        avg = self.compute_avg()
        return (
            f"Average Rouge-1 (F): {avg['rouge1']:.4f}\n"
            f"Average Rouge-2 (F): {avg['rouge2']:.4f}\n"
            f"Average Rouge-L (F): {avg['rougeL']:.4f}\n"
            f"Average Token Generate Sum: {avg['token_generate_sum']:.4f}\n"
            f"Average Skip Layer: {avg['avg_skip_layer']:.4f}\n"
            f"Average Infer Time: {avg['time']:.4f}\n"
        )

    def save_output_csv(self, path):
        """按顺序保存每条样本的预测与参考"""
        df = pd.DataFrame({
            "prediction": self.output,
            "reference": self.ref,
            "rouge1": self.rouge1_list,
            "rouge2": self.rouge2_list,
            "rougeL": self.rougeL_list,
        })

        df.to_csv(path, index=False, encoding="utf-8")
        print(f"[Saved] Output saved to {path}, total {len(df)} samples.")


class LayerwiseHiddenDiffCollector_full:
    def __init__(self):
        self.records = []
    
    def compute_metric(self, h_in, h_out):
        h_in_last = h_in
        h_out_last = h_out

        l2 = torch.norm(h_out_last - h_in_last, p=2).item()
        cos = F.cosine_similarity(h_in_last, h_out_last, dim=-1).mean().item()

        # 相对幅度 Relative Magnitude
        h_in_norm = torch.norm(h_in_last, p=2).item()
        rel_mag = l2 / (h_in_norm)  

        # if l2>=self.max_l2:
        #     self.max_l2 = l2

        # index = self.compute_cos_l2(cos, l2)

        return l2, cos, rel_mag

    def add_records(self, mode, layer_idx, token_str, l2, cos, rel_mag=0):
        self.records.append({
            "mode": mode,
            "layer": layer_idx,
            "token": token_str,
            "l2": l2,
            "cos": cos,
            "rel_mag": rel_mag
        })

class LayerwiseHiddenDiffCollector_skip:
    def __init__(self):
        self.records = []

    def compute(self, mode, layer_idx, h_in, h_out, token_str):
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

        self.records.append(
            {
                "mode": mode,
                "layer": layer_idx,
                "token": token_str,
                "l2": l2,
                "cos": cos,
                "rel_mag": rel_mag
            }
        )  

        LRS = 0.5*(1-cos) + 0.5*(rel_mag)
        return LRS