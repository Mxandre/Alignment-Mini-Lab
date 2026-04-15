import torch
from torch.nn.utils.rnn import pad_sequence

class ExperienceBuffer:
    def __init__(self):
        self.queries = []
        self.responses = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.advantages = []
        self.returns = []
        self.ref_log_prob = []

    def add(self, query, response, logprob, value, reward):
        """
        每采样一个 sample，存入 buffer
        """
        self.queries.append(query)       # [prompt_len]
        self.responses.append(response) # [response_len]
        self.logprobs.append(logprob)   # [response_len]
        self.values.append(value)       # [total_len]
        self.rewards.append(reward)     # scalar (score from RM)
    
    def add_adv_ret(self, advantages, returns, ref_log_prob) :
        self.advantages.append(advantages)
        self.returns.append(returns)
        self.ref_log_prob.append(ref_log_prob)

    def clear(self):
        self.queries = []
        self.responses = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.advantages = []
        self.returns = []
        self.ref_log_prob = []

    def get_all(self):
        """
        将列表转换为 Tensor，方便 Trainer 计算
        注意：实际操作中需要对不同长度的序列进行 Padding
        """
        return {
            "queries": self.queries,
            "responses": self.responses,
            "logprobs": torch.stack(self.logprobs),
            "values": torch.stack(self.values),
            "rewards": torch.stack(self.rewards),
        }

    # ===== Codex added: pad variable-length samples only when building a training batch =====
    def _to_1d_tensor(self, tensor):
        """
        Codex added:
        Normalize a single-sample sequence into shape [seq_len].
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)

        if tensor.dim() == 0:
            return tensor

        if tensor.dim() == 2 and tensor.size(0) == 1:
            return tensor.squeeze(0)

        if tensor.dim() != 1:
            raise ValueError(f"Expected a 1D tensor for a single sample, got {tuple(tensor.shape)}")

        return tensor

    def _left_pad_sequences(self, sequences, pad_token_id):
        """
        Codex added:
        Left-pad query tensors so the real prompt tokens stay adjacent to the response
        after concatenation.
        """
        max_len = max(x.size(0) for x in sequences)
        batch_size = len(sequences)
        padded = torch.full(
            (batch_size, max_len),
            fill_value=pad_token_id,
            dtype=sequences[0].dtype,
        )

        for i, seq in enumerate(sequences):
            padded[i, max_len - seq.size(0) :] = seq

        return padded

    def get_padded_batch(self, pad_token_id, device=None):
        """
        Codex added:
        Convert the variable-length samples stored in buffer into a padded batch.

        Expected per-sample format:
        - query: [query_len]        (unpadded)
        - response: [response_len]  (unpadded)
        - logprob: [response_len]
        - value: [response_len]
        - reward: scalar
        """
        if len(self.queries) == 0:
            raise ValueError("ExperienceBuffer is empty.")

        queries = [self._to_1d_tensor(x).long() for x in self.queries]
        responses = [self._to_1d_tensor(x).long() for x in self.responses]
        logprobs = [self._to_1d_tensor(x).float() for x in self.logprobs]
        values = [self._to_1d_tensor(x).float() for x in self.values]
        rewards = [self._to_1d_tensor(x).float() for x in self.rewards]
        advantages = [self._to_1d_tensor(x).float() for x in self.advantages]
        returns = [self._to_1d_tensor(x).float() for x in self.returns]
        ref_log_prob = [self._to_1d_tensor(x).float() for x in self.ref_log_prob]

        query_lens = torch.tensor([x.numel() for x in queries], dtype=torch.long)
        response_lens = torch.tensor([x.numel() for x in responses], dtype=torch.long)

        padded_queries = self._left_pad_sequences(queries, pad_token_id)
        padded_responses = pad_sequence(responses, batch_first=True, padding_value=pad_token_id)
        padded_logprobs = pad_sequence(logprobs, batch_first=True, padding_value=0.0)
        padded_values = pad_sequence(values, batch_first=True, padding_value=0.0)
        padded_advantages = pad_sequence(advantages, batch_first=True, padding_value=0.0)
        padded_returns = pad_sequence(returns, batch_first=True, padding_value=0.0)
        padded_ref_log_prob = pad_sequence(ref_log_prob, batch_first=True, padding_value=0.0)
        rewards = torch.stack(rewards).view(-1)

        full_input_ids = torch.cat([padded_queries, padded_responses], dim=1)
        attention_mask = full_input_ids.ne(pad_token_id).long()

        response_mask = torch.zeros_like(full_input_ids, dtype=torch.long)
        query_pad_len = padded_queries.size(1)
        for i, res_len in enumerate(response_lens.tolist()):
            response_mask[i, query_pad_len : query_pad_len + res_len] = 1

        response_token_mask = padded_responses.ne(pad_token_id).long()

        batch = {
            "queries": padded_queries,
            "responses": padded_responses,
            "logprobs": padded_logprobs,
            "values": padded_values,
            "rewards": rewards,
            "query_lens": query_lens,
            "query_pad_len": torch.tensor(query_pad_len, dtype=torch.long),
            "response_lens": response_lens,
            "full_input_ids": full_input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "response_token_mask": response_token_mask,
            "advantages": padded_advantages,
            "returns": padded_returns,
            "ref_log_prob" : padded_ref_log_prob
        }

        if device is not None:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }

        return batch
