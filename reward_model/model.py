import torch
import torch.nn as nn
from transformers import AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


class MyRewardModel(nn.Module):
    def __init__(self, model_name_or_path, r, alpha, dropout):
        super().__init__()
        # bnb_config = BitsAndBytesConfig(
        #     load_in_8bit=True
        # )
        # backbone = AutoModel.from_pretrained(model_name_or_path, quantization_config = bnb_config, dtype = torch.bfloat16, device_map = "cuda")
        backbone = AutoModel.from_pretrained(model_name_or_path, dtype = torch.bfloat16, device_map = "cuda")
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
        peft_config = LoraConfig(
            task_type = TaskType.FEATURE_EXTRACTION,
            r = r, 
            lora_alpha = alpha,
            lora_dropout = dropout,
            target_modules = ["q_proj", "v_proj", "k_proj","o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.backbone = get_peft_model(backbone, peft_config)
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()

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