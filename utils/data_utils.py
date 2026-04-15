from utils.config import PPOTrainConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
def build_prompt_dataset(config : PPOTrainConfig) :
    if config.dataset_name == "json":
        raw_dataset = load_dataset(
            "json",
            data_files={"train": config.train_file, "eval": config.eval_file},
        )
    else:
        raw_dataset = load_dataset(config.dataset_name)
    def extract_prompt(example):
        return {"prompt": example[config.prompt_key]}
    column_names = raw_dataset["train"].column_names
    train_dataset = raw_dataset["train"].map(
        extract_prompt,
        remove_columns=column_names,
    )
    eval_dataset = raw_dataset["eval"].map(
        extract_prompt, 
        remove_columns=column_names
    )
    train_dataset = train_dataset.filter(lambda x: len(x["prompt"]) > 5)
    eval_dataset = eval_dataset.filter(lambda x: len(x["prompt"]) > 5)
    return train_dataset, eval_dataset

def prompt_collate_fn(batch, config : PPOTrainConfig, tokenizer):
    prompts = [x["prompt"]  for x in batch]
    tokenized = tokenizer(
        prompts,
        return_tensors = "pt",
        padding = True,
        truncation = True,
        max_length = config.max_prompt_length
    )
    return {
        "prompts" : prompts,
        "input_ids" : tokenized["input_ids"],
        "attention_mask" : tokenized["attention_mask"]
    }


def build_dataloader(config : PPOTrainConfig, tokenizer):
    train_dataset, eval_dataset = build_prompt_dataset(config)
    train_dataloader =  DataLoader(
            train_dataset,
            batch_size = config.rollout_batch_size,
            shuffle=True,
            collate_fn=lambda x : prompt_collate_fn(x, config, tokenizer)
        )
    eval_dataloader =  DataLoader(
            eval_dataset,
            batch_size = config.eval_batch_size,
            shuffle=False,
            collate_fn=lambda x : prompt_collate_fn(x, config, tokenizer)
        )
    return train_dataloader, eval_dataloader