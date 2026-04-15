import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType

class MyRewardModel(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        backbone = AutoModel.from_pretrained(model_name_or_path, dtype = torch.bfloat16)
        
        peft_config = LoraConfig(
            task_type = TaskType.FEATURE_EXTRACTION,
            r = 8, 
            lora_alpha = 16,
            lora_dropout = 0.1,
            target_modules = ["q_proj", "v_proj", "k_proj","o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.backbone = get_peft_model(backbone, peft_config)

        self.config = self.backbone.config
        hidden_size = getattr(self.config, "word_embed_proj_dim", self.config.hidden_size)
        
        self.value_head = nn.Linear(hidden_size, 1, bias = False, dtype = torch.bfloat16)



    def forward(self, input_ids, attention_mask):
        transformer_outputs = self.backbone(input_ids, attention_mask)
        hidden_states = transformer_outputs.last_hidden_state

        last_token_index = torch.sum(attention_mask, dim = 1) - 1 ## right padding
        last_token_hidden = torch.gather(hidden_states, 1, last_token_index.view(-1, 1, 1).expand((-1, 1, hidden_states.size(-1))))

        rewards = self.value_head(last_token_hidden)

        return rewards.view(-1)