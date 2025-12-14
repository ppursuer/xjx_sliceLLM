from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

class ModelName:
    LLAMA2_7B: str = "llama2-7b"
    LLAMA2_13B: str = "llama2-13b"
    LLAMA3_8B: str = "llama3-8b"

class ModelPath:
    # LLAMA2_7B: str = "/mnt/Data/xjx/model/hf-model/Llama-2-7b-hf"
    LLAMA2_7B: str = "/root/autodl-tmp/models"
    # LLAMA2_13B: str = "/mnt/Data/xjx/model/hf-model/Llama-2-13b-hf"
    LLAMA2_13B: str = "/data/xjx/llama2-13b/models"
    LLAMA3_8B: str = "/mnt/Data/xjx/model/hf-model/llama3-8b"

def get_tokenizer_and_model(model_name, card):
    device = torch.device(f"cuda:{card}" if torch.cuda.is_available() else "cpu")
    
    if model_name == ModelName.LLAMA2_7B:
        path = ModelPath.LLAMA2_7B
    elif model_name == ModelName.LLAMA2_13B:
        path = ModelPath.LLAMA2_13B
    elif model_name == ModelName.LLAMA3_8B:
        path = ModelPath.LLAMA3_8B
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 避免没有pad token报错
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        ).to(device)
    model.eval()

    return tokenizer, model, device 

def pre_setup_skip(model_name):
    if model_name == ModelName.LLAMA2_13B:
        return 37, 37, 25
    elif model_name == ModelName.LLAMA2_7B:
        return 29, 27, 22  # 目前效果最好