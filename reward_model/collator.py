from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from typing import List, Dict, Any
import torch

@dataclass
class MyRewardCollator :
    tokenizer: Any
    max_length: int = 512

    def __post_init__(self):
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos.token
        self.tokenizer.padding_side = "right"
    
    def __call__(self, features : List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        all_chosen_texts = []
        all_rejected_texts = []

        for item in features :
            p = item["prompt"]
            all_chosen_texts.append(f"Prompt: {p}\n\nResponse: {item['chosen']}")
            all_rejected_texts.append(f"Prompt: {p}\n\nResponse: {item['rejected']}")\
            
        chosen_batch = self.tokenizer(
            all_chosen_texts,
            max_length = self.max_length,
            truncation = True,
            padding = True,
            return_tensors = "pt"
        )

        rejected_batch = self.tokenizer(
            all_rejected_texts,
            max_length = self.max_length,
            truncation = True,
            padding = True,
            return_tensors = "pt"
        )

        return {
            "chosen_input_ids" : chosen_batch["input_ids"],
            "chosen_attention_mask" : chosen_batch["attention_mask"],
            "rejected_input_ids" : rejected_batch["input_ids"],
            "rejected_attention_mask" :  rejected_batch["attention_mask"]
        }
    
class SimpleInferenceCollator:
    def __init__(self, tokenizer, max_length = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        inputs = self.tokenizer(
            batch,
            padding = True,
            truncation = True,
            max_length = self.max_length,
            return_tensors = "pt"
        )
        return {
            "input_ids" : inputs["input_ids"],
            "attention_mask" : inputs["attention_mask"]
        }