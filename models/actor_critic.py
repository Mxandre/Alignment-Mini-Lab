import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel
import torch 
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
class ActorModel(nn.Module):
    def __init__(self, model_name, lora_config = None):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.bfloat16, device_map = "cuda")
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r = lora_config["r"],
            lora_alpha= lora_config["alpha"],
            target_modules=lora_config["modules"]
        )
        self.backbone = get_peft_model(model, lora_config)

    def forward(self, input_ids, attention_mask = None) :
        outputs = self.backbone(input_ids, attention_mask)
        return outputs.logits
    
    @torch.no_grad()
    def generate(self, input_ids, attention_mask = None, **gen_kwargs):
        outputs = self.backbone.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            **gen_kwargs
        )
        return outputs
    
    def get_log_probs(self, logits, labels, query_len):
        log_probs = torch.log_softmax(logits, dim = -1)
        shift_logits = logits[:, query_len-1:-1, :].contiguous()
        shift_lables = labels[:, query_len:].contiguous()

        log_probs = torch.log_softmax(shift_logits, dim = -1)
        per_token_log_probs = torch.gather(log_probs, dim = 2, index = shift_lables.unsqueeze(-1)).squeeze(-1)
        return per_token_log_probs

    # ===== Codex added: compute response-token logprobs from full [query + response] inputs =====
    def get_response_log_probs_from_full_inputs_v2(
        self,
        logits, ## logits都是transformer返回的结果
        full_input_ids,
        query_pad_lens,
        response_lens,
        pad_value=0.0,
    ):
        """
        Codex added:
        Extract per-token log probs only for the response span of each sample.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        per_sample_log_probs = []
        q_len = query_pad_lens

        batch_size = full_input_ids.size(0)
        for i in range(batch_size):
            r_len = int(response_lens[i].item())

            token_log_probs = log_probs[i, q_len - 1 : q_len + r_len - 1, :]
            token_labels = full_input_ids[i, q_len : q_len + r_len]
            token_log_probs = torch.gather(
                token_log_probs,
                dim=-1,
                index=token_labels.unsqueeze(-1),
            ).squeeze(-1)
            per_sample_log_probs.append(token_log_probs)

        return pad_sequence(per_sample_log_probs, batch_first=True, padding_value=pad_value)
    
class CriticModel(nn.Module):
    def __init__(self, model_name, lora_config = None):
        super().__init__()
        model = AutoModel.from_pretrained(model_name, torch_dtype = torch.bfloat16, device_map = "cuda")
        self.hidden_size = model.config.hidden_size
        self.value_head = nn.Linear(self.hidden_size, 1, dtype = torch.bfloat16).to("cuda")
        lora_config = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            r = lora_config["r"],
            lora_alpha= lora_config["alpha"],
            target_modules=lora_config["modules"]
        )
        self.backbone = get_peft_model(model, lora_config)

    def forward(self, input_ids, attention_mask = None):
        transformer_output = self.backbone(input_ids, attention_mask, output_hidden_states=True)
        last_hidden_state = transformer_output.last_hidden_state
        values = self.value_head(last_hidden_state) 
        return values.squeeze(-1)
    
    def get_res_values(self, values, query_len):
        return values[:, query_len - 1 : -1]

    # ===== Codex added: align critic values with response tokens from full inputs =====
    def get_response_values_from_full_values(
        self,
        values,
        query_lens,
        response_lens,
        pad_value=0.0,
    ):
        """
        Codex added:
        The value at position q_len - 1 is aligned with the first response token.
        """
        per_sample_values = []
        q_len = query_lens

        batch_size = values.size(0)
        for i in range(batch_size):
            r_len = int(response_lens[i].item())
            token_values = values[i, q_len - 1 : q_len + r_len - 1]
            per_sample_values.append(token_values)

        return pad_sequence(per_sample_values, batch_first=True, padding_value=pad_value)

    # ===== Codex added: v2 helpers for left-padded query block + right-padded response block =====
    def get_response_log_probs_from_full_inputs_v2(
        self,
        logits,
        full_input_ids,
        query_pad_len,
        response_lens,
        pad_value=0.0,
    ):
        """
        Codex added:
        queries are left-padded to query_pad_len, and responses start exactly at
        index query_pad_len.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        per_sample_log_probs = []
        q_pad_len = int(query_pad_len.item()) if isinstance(query_pad_len, torch.Tensor) else int(query_pad_len)

        batch_size = full_input_ids.size(0)
        for i in range(batch_size):
            r_len = int(response_lens[i].item())
            token_log_probs = log_probs[i, q_pad_len - 1 : q_pad_len + r_len - 1, :]
            token_labels = full_input_ids[i, q_pad_len : q_pad_len + r_len]
            token_log_probs = torch.gather(
                token_log_probs,
                dim=-1,
                index=token_labels.unsqueeze(-1),
            ).squeeze(-1)
            per_sample_log_probs.append(token_log_probs)

        return pad_sequence(per_sample_log_probs, batch_first=True, padding_value=pad_value)

    def get_response_values_from_full_values_v2(
        self,
        values,
        query_pad_len,
        response_lens,
        pad_value=0.0,
    ):
        """
        Codex added:
        Align critic values to response tokens when the query block is left-padded.
        """
        per_sample_values = []
        q_pad_len = int(query_pad_len.item()) if isinstance(query_pad_len, torch.Tensor) else int(query_pad_len)

        batch_size = values.size(0)
        for i in range(batch_size):
            r_len = int(response_lens[i].item())
            token_values = values[i, q_pad_len - 1 : q_pad_len + r_len - 1]
            per_sample_values.append(token_values)

        return pad_sequence(per_sample_values, batch_first=True, padding_value=pad_value)


        
