import argparse
import torch
import time
import os
import numpy as np
import time

from torchmetrics.text import ROUGEScore
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt

from get_data import get_data
from model_and_tokenizer import get_tokenizer_and_model, pre_setup_skip

from metric_collector import Metric_Skip, LayerwiseHiddenDiffCollector_skip
from inference import forward_skip

"""
skip 
llama2-7b 限制 （22层-29层】   上限是7层
llama2-13b ：限制（25层-37层】 上限是12层
"""

# class Metric_Skip:
#     def __init__(self, rouge_keys=("rouge1", "rouge2", "rougeL"), use_stemmer=True):
#         self.rouge_metric = ROUGEScore(rouge_keys=rouge_keys, use_stemmer=use_stemmer)
#         self.rouge1_list = []
#         self.rouge2_list = []
#         self.rougeL_list = []
#         self.skip_layer_list = []
#         self.infer_time_list = []
#         self.output = []
#         self.ref = []
        
#         self.LRS_dict = {}

#     def add(self, pred, ref, avg_skip_layer, infer_time):
#         """添加一条样本的 ROUGE"""
#         r1, r2, rL = self.compute_single(pred, ref)
#         self.rouge1_list.append(r1)
#         self.rouge2_list.append(r2)
#         self.rougeL_list.append(rL)
#         self.skip_layer_list.append(avg_skip_layer)
#         self.infer_time_list.append(infer_time)
#         self.ref.append(ref)
#         self.output.append(pred)
#         return r1, r2, rL
    
#     def compute_single(self, pred, ref):
#         scores = self.rouge_metric(preds=[pred], target=[ref])
#         r1 = scores["rouge1_fmeasure"].item()
#         r2 = scores["rouge2_fmeasure"].item()
#         rL = scores["rougeL_fmeasure"].item()
#         return r1, r2, rL
    
#     def compute_avg(self):
#         return {
#             "rouge1": np.mean(self.rouge1_list),
#             "rouge2": np.mean(self.rouge2_list),
#             "rougeL": np.mean(self.rougeL_list),
#             "avg_skip_layer": np.mean(self.skip_layer_list),
#             "time": np.mean(self.infer_time_list),
#         }
    
#     def summary_formatted(self):
#         avg = self.compute_avg()
#         return (
#             f"Average Rouge-1 (F): {avg['rouge1']:.4f}\n"
#             f"Average Rouge-2 (F): {avg['rouge2']:.4f}\n"
#             f"Average Rouge-L (F): {avg['rougeL']:.4f}\n"
#             f"Average Skip Layer: {avg['avg_skip_layer']:.4f}\n"
#             f"Average Infer Time: {avg['time']:.4f}\n"
#         )

# class LayerwiseHiddenDiffCollector_skip:
#     def __init__(self):
#         self.records = []

#     def compute(self, mode, layer_idx, h_in, h_out, token_str):
#         if torch.equal(h_in, h_out):
#             self.records.append(self.records[-1].copy())
#             return 
        
#         h_in_last = h_in
#         h_out_last = h_out

#         l2 = torch.norm(h_out_last - h_in_last, p=2).item()
#         cos = F.cosine_similarity(h_in_last, h_out_last, dim=-1).mean().item()

#         # 相对幅度 Relative Magnitude
#         h_in_norm = torch.norm(h_in_last, p=2).item()
#         rel_mag = l2 / (h_in_norm)  

#         self.records.append(
#             {
#                 "mode": mode,
#                 "layer": layer_idx,
#                 "token": token_str,
#                 "l2": l2,
#                 "cos": cos,
#                 "rel_mag": rel_mag
#             }
#         )  

#         LRS = 0.5*(1-cos) + 0.5*(rel_mag)
#         return LRS
# def forward_skip(model, input_ids, tokenizer, collector_skip, start_layer, end_layer, min_start_layer):
#     seq_len = input_ids.shape[1]
#     position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
#     hidden_states = model.model.embed_tokens(input_ids)

#     return_layer = start_layer
#     once = False
#     ################ 浅层 #################
#     for i in range(start_layer):
#         if i == return_layer:
#             break

#         layer = model.model.layers[i]
#         h_in = hidden_states[:, -1, :].detach()
#         hidden_states = layer(
#             hidden_states,
#             attention_mask=None,
#             position_ids=position_ids,
#             use_cache=False,
#         )[0]
#         h_out = hidden_states[:, -1, :].detach()

#         logits = model.lm_head(model.model.norm(h_out))
#         token_str = tokenizer.decode([torch.argmax(logits, -1).item()])
#         LRS = collector_skip.compute("skip", i, h_in, h_out, token_str)

#         # 决策跳跃 
#         # if LRS <= 0.2 and once is False: 
#         #     once = True 
#         #     return_layer = start_layer - 1 
#         #     start_layer = return_layer

#         if (LRS <= 0.2) and (not once):

#             once = True
#             # -------------------------
#             # 降低 start_layer
#             # -------------------------
#             dynamic_start = max(start_layer - 1, min_start_layer)

#             # 更新跳跃区间
#             start_layer = dynamic_start

#             return_layer = dynamic_start  # for log/debug

#     ################ skip layer #################  
#     for i in range(start_layer, end_layer):
#         collector_skip.compute("skip", i, h_in, h_out, token_str)    

#     #################  deep layer ################

#     for i in range(end_layer, len(model.model.layers)):
#         layer = model.model.layers[i]
#         h_in = hidden_states[:, -1, :].detach()

#         hidden_states = layer(
#             hidden_states,
#             attention_mask=None,
#             position_ids=position_ids,
#             use_cache=False,
#         )[0]

#         h_out = hidden_states[:, -1, :].detach()
#         logits = model.lm_head(model.model.norm(h_out))
#         token_str = tokenizer.decode([torch.argmax(logits, -1).item()])
#         collector_skip.compute("skip", i, h_in, h_out, token_str)
#     final_logits = model.lm_head(model.model.norm(hidden_states[:, -1, :]))
#     probs = torch.softmax(final_logits, dim=-1)
#     max_prob, next_token_id = torch.max(probs, dim=-1)

#     # confidence = max_prob.item()  
#     input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
#     return next_token_id, token_str, input_ids, return_layer, end_layer - start_layer


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
                next_token_id, next_token_str, input_ids, start_layer, skip_sum, infer_time = forward_skip(model,
                                                            input_ids, 
                                                            tokenizer, 
                                                            collector_skip,
                                                            start_layer,
                                                            end_layer,
                                                            args.min_start_layer,
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

    # 构建日志文件路径
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
        + metric_skip.summary_formatted() + 
        "=======================================================\n"
    )
    print(summary_text)
    # 写入日志文件（追加模式）
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Evaluation results saved to: {log_path}")

    # 保存整个数据集上的skip inference输出结果
    csv_path = os.path.join(log_dir, f"output.csv")
    metric_skip.save_output_csv(csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index (e.g. 0,1,2,3)")
    parser.add_argument("--model_name", type=str, default="llama2-7b", help="Path to the model")
    parser.add_argument("--dataset_name", type=str, default="xsum_summarization", help="Dataset to evaluate")
    parser.add_argument("--samples", type=int, default=10, help="sample to evaluate")
    parser.add_argument("--shots", type=int, default=1, help="n-shot")
    parser.add_argument("--infer_strategy", type=str, default="Dynamic_Skip_Inf", help="full_inf and skip_inf")
    parser.add_argument("--max_generate_token", type=int, default=60, help="生成token数量")
    args = parser.parse_args()
    args.end_layer,args.start_layer,args.min_start_layer = pre_setup_skip(args.model_name)
    main(args)
