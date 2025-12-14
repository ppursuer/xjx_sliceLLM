import argparse
import torch
import numpy as np
from torchmetrics.text import ROUGEScore
from torchmetrics.text import BLEUScore
from tqdm import tqdm

from get_data import get_data
from model_and_tokenizer import get_tokenizer_and_model

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

"""
motivation 1.2：
llama2-7b
跳跃{15-29}层 vs 全层
观察每层的输出情况

"""

class Metric:
    def __init__(self, rouge_keys=("rouge1", "rouge2", "rougeL"), use_stemmer=True, bleu_n_gram=4, smooth=False):
        self.rouge_metric = ROUGEScore(rouge_keys=rouge_keys, use_stemmer=use_stemmer)
        self.bleu_metric = BLEUScore(n_gram=bleu_n_gram, smooth=smooth)

    def compute_single(self, pred, ref):
        scores = self.rouge_metric(preds=[pred], target=[ref])
        r1 = scores["rouge1_fmeasure"].item()
        r2 = scores["rouge2_fmeasure"].item()
        rL = scores["rougeL_fmeasure"].item()

        jac = self.compute_jaccard(pred, ref)
        # bleu = self.compute_bleu_single(pred, ref)
        return r1, r2, rL, jac

    def compute_jaccard(self, pred, ref):
        """
        计算 Jaccard 相似度（基于 token 集合）
        """
        pred_tokens = set(pred.strip().split())
        ref_tokens = set(ref.strip().split())

        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0

        intersection = len(pred_tokens & ref_tokens)
        union = len(pred_tokens | ref_tokens)

        return intersection / union

    def compute_bleu_single(self, pred, ref):
        """
        计算单个样本的BLEU指标

        Args:
            pred: 模型生成的预测文本
            ref: 参考文本

        Returns:
            float: BLEU分数（范围[0, 1]）
        """
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()  # BLEU指标要求参考文本是列表的列表
        
        # 处理极端情况（预测文本为空词列表）
        if len(pred_tokens) == 0:
            return 0.0
        
        # 计算BLEU分数
        bleu_score = self.bleu_metric(
            preds = [pred_tokens],
            target = [[ref_tokens]]
        )
        return bleu_score.item()

def forward_full(model, input_ids, tokenizer):
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
        token_str = tokenizer.decode([torch.argmax(logits, -1).item()])

        # collector_full.add("full", i, h_in, h_out, token_str)

    final_logits = model.lm_head(model.model.norm(h_out))
    return final_logits

def forward_skip(model, input_ids, tokenizer, SKIP_START, SKIP_END):
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

    hidden_states = model.model.embed_tokens(input_ids)

    for i, layer in enumerate(model.model.layers):
        h_in = hidden_states[:, -1, :].detach()
        # print(h_in)
        if SKIP_START <= i <= SKIP_END:
            # 跳过层：不更新
            h_out = hidden_states[:, -1, :].detach()
            # print(h_out)
        else:
            # 正常前向
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                use_cache=False
            )[0]
            h_out = hidden_states[:, -1, :].detach()
            # print(h_out)

        # 当前层 top1 token
        logits = model.lm_head(model.model.norm(h_out))
        token_str = tokenizer.decode([torch.argmax(logits, -1).item()])

        # collector_skip.add("skip", i, h_in, h_out, token_str)

    final_logits = model.lm_head(model.model.norm(h_out))
    return final_logits

def main(args):
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if args.model_name == "llama2-7b":
        start_pos_list = [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15]
        end_pos = 29
    elif args.model_name == "llama2-13b":
        start_pos_list = [37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23]
        end_pos = 37

    stop_tokens = [
        ".", "!", "?",          # 句子终止符
        "\n", "\t",             # 换行 / tab
    ]

    evaluation_dataset = get_data(
        args.dataset_name,
        samples=args.samples,
        shots=args.shots,
    )

    tokenizer, model, device = get_tokenizer_and_model(args.model_name, args.cuda)

    for i, data_point in enumerate(tqdm(evaluation_dataset, desc="Evaluating")):
        input_text = data_point.input
        ref_text = data_point.ref

        input_ids_full = tokenizer.encode(input_text, return_tensors="pt").to(device)
        input_ids_skip = tokenizer.encode(input_text, return_tensors="pt").to(device)
        input_len = input_ids_full.shape[1]

        metric = Metric()
        output_list = []
        for token_iter in range(args.max_generate_token):
            # collector_full = LayerwiseHiddenDiffCollector()  

            ################# full FORWARD ################# 
            full_logits = forward_full(model,
                                       input_ids_full, 
                                       tokenizer, 
                                       )
            probs = torch.softmax(full_logits, dim=-1)
            max_prob, next_token_id = torch.max(probs, dim=-1)

            confidence = max_prob.item()  
            input_ids_full = torch.cat([input_ids_full, next_token_id.unsqueeze(0)], dim=-1)
            # print(f"FULL FORWARD Token {token_iter+1} | Conf: {confidence:.4f} | Token: {tokenizer.decode([next_token_id.item()])}")
  
        generated_text = tokenizer.decode(input_ids_full[0, input_len:], skip_special_tokens=True)     
        # print(generated_text)

        ################# skip FORWARD ################# 
        for _,start_pos in enumerate(start_pos_list):
            input_ids_skip_copy = input_ids_skip.clone()
            for token_iter in range(args.max_generate_token):
                skip_logits = forward_skip(model,
                                            input_ids_skip_copy, 
                                            tokenizer, 
                                            start_pos,
                                            end_pos,
                                            )
                probs = torch.softmax(skip_logits, dim=-1)
                max_prob, next_token_id = torch.max(probs, dim=-1)
                next_token_str = tokenizer.decode([torch.argmax(skip_logits, -1).item()])
                # confidence = max_prob.item()
                input_ids_skip_copy = torch.cat([input_ids_skip_copy, next_token_id.unsqueeze(0)], dim=-1)
                if next_token_id == tokenizer.eos_token_id or any(s in next_token_str for s in stop_tokens):
                    break
                # print(f"SKIP FORWARD Token {token_iter+1} | Conf: {confidence:.4f} | Token: {tokenizer.decode([next_token_id.item()])}")
            
            # print(tokenizer.decode(input_ids_skip_copy[0, input_len:], skip_special_tokens=True))
            output_list.append(tokenizer.decode(input_ids_skip_copy[0, input_len:], skip_special_tokens=True))
    
        print(f"   Full inference   :   {tokenizer.decode(input_ids_full[0, input_len:], skip_special_tokens=True)}")

    # metric = Metric()
        for i,content in enumerate(output_list):
            print(f"Skip at {start_pos_list[i]} inference:   {content}")
            # print("指标")
            r1, r2, rl, jac = metric.compute_single(ref_text, content)
            # print("rouge-1:", r1)
            # print("rouge-2:", r2)
            # print("rouge-L:", rl)
            # print("JAC:", jac)
            # print()
            r1, r2, rl, jac = metric.compute_single(generated_text, content)
            print("rouge-1:", r1)
            print("rouge-2:", r2)
            print("rouge-L:", rl)
            print("JAC:", jac)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index (e.g. 0,1,2,3)")
    parser.add_argument("--model_name", type=str, default="llama2-7b", help="Path to the model")
    parser.add_argument("--dataset_name", type=str, default="xsum_summarization", help="Dataset to evaluate")
    parser.add_argument("--samples", type=int, default=1, help="sample to evaluate")
    parser.add_argument("--shots", type=int, default=1, help="n-shot")
    parser.add_argument("--infer_strategy", type=str, default="Full inf", help="full_inf and mine_inf Static_skip")
    parser.add_argument("--max_generate_token", type=int, default=30, help="生成token数量")
    args = parser.parse_args()
    main(args)

