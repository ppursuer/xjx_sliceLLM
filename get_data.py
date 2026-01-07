import datasets
from dataclasses import dataclass

@dataclass
class EvaluationExample:
    input: str
    ref: str

class DatasetPath:
    # CNN_DM_SUMMARIZATION: str = "/mnt/Data/xjx/data/cnn_dailymail/3.0.0"
    CNN_DM_SUMMARIZATION: str = "/root/datasets/cnn"
    # XSUM_SUMMARIZATION: str = "/mnt/Data/xjx/data/xsum/data"
    XSUM_SUMMARIZATION: str = "/root/datasets/xsum"
    HUMAN_EVAL: str = "/root/datasets/human_eval"

class DatasetFormat:
    CNN_DM_SUMMARIZATION: str = "cnn_dm_summarization"
    XSUM_SUMMARIZATION: str = "xsum_summarization"
    HUMAN_EVAL: str = "human_eval"

class ModelTemplate:
    XSUM_SUMMARIZATION: str = "xsum_summarization"

def get_xsum_summarization_n_fewshot(path, n, shot):
    dataset = datasets.load_dataset(
        "parquet",
        data_files={
            "train": f"{path}/xsum_train.parquet",
            "test": f"{path}/xsum_test.parquet",
            "validation": f"{path}/xsum_validation.parquet"
        }
    )
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    prompt_keys=["document", "summary"]
  
    prompt_shots = ""
    prompt_item = train_dataset.shuffle(seed=42).select(range(shot))
    if shot > 0:
        for i in range(shot):
            prompt = f"""    
Document:{prompt_item[0][prompt_keys[0]]}
Summary:{prompt_item[0][prompt_keys[1]]}
            """
            prompt_shots += prompt
    # prompt_shots += "Now, Summarize the following article in one concise sentence."
    
    n_item = test_dataset.shuffle(seed=42).select(range(n))

    evaluation_dataset = []
    for item in n_item:
        document = item["document"]
        cons_text = item["summary"]   
        input_prompt = prompt_shots + f"\nDocument:{document}\nNow, Summarize the following article in one concise sentence.\nSummary:"
        # prompt_shots += "Now, Summarize the following article in one concise sentence."
    
        # print(input_prompt)
        evaluation_dataset.append(
            EvaluationExample(
                input=input_prompt,
                ref=f"{cons_text}",
            )
        )
    return evaluation_dataset

def get_xsum_summarization_all_fewshot(path, shot):
    dataset = datasets.load_dataset(
        "parquet",
        data_files={
            "train": f"{path}/xsum_train.parquet",
            "test": f"{path}/xsum_test.parquet",
            "validation": f"{path}/xsum_validation.parquet"
        }
    )
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    prompt_keys=["document", "summary"]

    prompt_shots = ""
    if shot > 0:
        prompt_item = train_dataset.shuffle(seed=42).select(range(shot))
        for i in range(shot):
            prompt = f"""    
Document:{prompt_item[0][prompt_keys[0]]}
Summary:{prompt_item[0][prompt_keys[1]]}
            """
            prompt_shots += prompt 

    evaluation_dataset = []
    for item in test_dataset:
        document = item["document"]
        cons_text = item["summary"]    
        input_prompt = prompt_shots + f"\nDocument:{document}\nNow, Summarize the following article in one concise sentence.\nSummary:"
        # print(input_prompt)
        evaluation_dataset.append(
            EvaluationExample(
                input=input_prompt,
                ref=f"{cons_text}",
            )
        )
    return evaluation_dataset

def get_cnn_dm_summarization_n_fewshot(path, n, shot):
    dataset = datasets.load_dataset(
        "parquet",
        data_files={
            "train": [
                f"{path}/train-00000-of-00003.parquet",
                f"{path}/train-00001-of-00003.parquet",
                f"{path}/train-00002-of-00003.parquet"
            ],
            "test": f"{path}/test-00000-of-00001.parquet",
            "validation": f"{path}/validation-00000-of-00001.parquet"
        }
    )
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    prompt_keys=["article", "highlights"]

    prompt_shots = ""
    if shot > 0:
        prompt_item = train_dataset.shuffle(seed=42).select(range(shot))
        for i in range(shot):
            prompt = f"""    
Article:{prompt_item[0][prompt_keys[0]]}
Highlights:{prompt_item[0][prompt_keys[1]]}
            """
            prompt_shots += prompt
    # prompt_shots += "Now, Summarize the following article in one concise sentence."
    
    n_item = test_dataset.shuffle(seed=42).select(range(n))

    evaluation_dataset = []
    for item in n_item:
        document = item["article"]
        cons_text = item["highlights"]   
        input_prompt = prompt + f"\nArticle:{document}\nHighlights:"
        # print(input_prompt)
        evaluation_dataset.append(
            EvaluationExample(
                input=input_prompt,
                ref=f"{cons_text}",
            )
        )
    return evaluation_dataset 

def get_cnn_dm_summarization_all_fewshot(path, shot):
    dataset = datasets.load_dataset(
        "parquet",
        data_files={
            "train": [
                f"{path}/train-00000-of-00003.parquet",
                f"{path}/train-00001-of-00003.parquet",
                f"{path}/train-00002-of-00003.parquet"
            ],
            "test": f"{path}/test-00000-of-00001.parquet",
            "validation": f"{path}/validation-00000-of-00001.parquet"
        }
    )
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    prompt_keys=["article", "highlights"]

    prompt_shots = ""
    if shot > 0:
        prompt_item = train_dataset.shuffle(seed=42).select(range(shot))
        for i in range(shot):
            prompt = f"""    
Article:{prompt_item[0][prompt_keys[0]]}
Highlights:{prompt_item[0][prompt_keys[1]]}
            """
            prompt_shots += prompt

    evaluation_dataset = []
    for item in test_dataset:
        document = item["article"]
        cons_text = item["highlights"]     
        input_prompt = prompt + f"\nArticle:{document}\nHighlights:"
        # print(input_prompt)
        evaluation_dataset.append(
            EvaluationExample(
                input=input_prompt,
                ref=f"{cons_text}",
            )
        )
    return evaluation_dataset 

def prepare_human_eval(path):
    dataset = datasets.load_dataset(
        "parquet",
        data_files={
            "test": f"{path}/test-00000-of-00001.parquet",
        }
    )
    test_dataset = dataset["test"]  

    evaluation_dataset = []
    for item in test_dataset:
        prompt = item["prompt"]
        evaluation_dataset.append(
            EvaluationExample(
                input=prompt,
                ref=item["canonical_solution"],
            )
        )
    return evaluation_dataset

def get_data(dataset_name, samples, shots):
    if dataset_name == DatasetFormat.CNN_DM_SUMMARIZATION:
        path = DatasetPath.CNN_DM_SUMMARIZATION
        if samples != -1:
            evaluation_dataset = get_cnn_dm_summarization_n_fewshot(path, samples, shots)
        else :
            evaluation_dataset = get_cnn_dm_summarization_all_fewshot(path, shots)
    
    elif dataset_name == DatasetFormat.XSUM_SUMMARIZATION:
        path = DatasetPath.XSUM_SUMMARIZATION
        if samples != -1:
            evaluation_dataset = get_xsum_summarization_n_fewshot(path, samples, shots)
        else :
            evaluation_dataset = get_xsum_summarization_all_fewshot(path, shots)

    elif dataset_name == DatasetFormat.HUMAN_EVAL:
        path = DatasetPath.HUMAN_EVAL
        evaluation_dataset = prepare_human_eval(path)

    return evaluation_dataset


""""
文本摘要类数据集: 比拼输出质量（在一定输出内容范围内）
xsum数据集: 一句话回答

cnn数据集: 三句话回答



GSM8K数据集: 

"""