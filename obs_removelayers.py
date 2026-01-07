

"""
obs remove per layer 的影响
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
    parser.add_argument("--max_generate_token", type=int, default=60, help="最大生成token数量")
    
    args = parser.parse_args()
    main(args)
