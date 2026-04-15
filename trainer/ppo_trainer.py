import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import tqdm
import wandb
from utils.config import PPOTrainConfig
from models.actor_critic import ActorModel, CriticModel

def compute_advantages(values, rewards, mask, gamma, lam):
    advantages = torch.zeros_like(rewards)

    seq_len = rewards.size(1)
    last_gae_lam = 0

    for t in reversed(range(seq_len)):
        next_value = values[:, t+1] if t < seq_len else 0

        delta = rewards[:, t] + gamma * next_value - values[:,t]

        advantages[:, t] = delta + gamma*lam*last_gae_lam
        last_gae_lam = advantages[:, t]
    
    returns = advantages + values
    advantages = (advantages - advantages.mean())/ (advantages.std + 1e-8)
    return advantages, returns

def get_kl_penalty(log_probs, ref_log_probs, kl_coef = 0.1):
    kl = log_probs - ref_log_probs
    return -kl_coef * kl

def prepare_inputs(queries, responses, pad_token_id):

    full_input_ids = torch.cat([queries, responses], dim=1)
    

    attention_mask = full_input_ids.ne(pad_token_id).long()
    
    response_mask = torch.zeros_like(full_input_ids)
    query_len = queries.size(1)
    for i in range(len(responses)):
        # 找到非 padding 的长度
        res_len = responses[i].ne(pad_token_id).sum()
        response_mask[i, query_len : query_len + res_len] = 1
        
    return full_input_ids, attention_mask, response_mask

# ===== Codex added: helpers for padded variable-length PPO batches =====
def masked_mean(values, mask, eps=1e-8):
    mask = mask.float()
    return (values * mask).sum() / mask.sum().clamp_min(eps)

def whiten_masked(values, mask, eps=1e-8):
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    mean = (values * mask).sum() / denom
    var = (((values - mean) * mask) ** 2).sum() / denom
    return (values - mean) / torch.sqrt(var + eps)

def build_token_level_rewards(rewards, kl_rewards, response_token_mask):
    """
    Codex added:
    Use token-level KL reward everywhere, then add the scalar RM reward
    onto the last valid response token.
    """
    token_rewards = kl_rewards * response_token_mask.float()
    response_lens = response_token_mask.sum(dim=1).long()

    for i, res_len in enumerate(response_lens.tolist()):
        if res_len > 0:
            token_rewards[i, res_len - 1] = token_rewards[i, res_len - 1] + rewards[i]

    return token_rewards

def compute_advantages_v2(values, rewards, mask, gamma, lam):
    """
    Codex added:
    GAE over padded response-token tensors of shape [batch, max_response_len].
    """
    mask = mask.float()
    advantages = torch.zeros_like(rewards)
    last_gae_lam = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)
    seq_len = rewards.size(1)

    for t in reversed(range(seq_len)):
        next_values = values[:, t + 1] if t < seq_len - 1 else torch.zeros_like(last_gae_lam)
        next_mask = mask[:, t + 1] if t < seq_len - 1 else torch.zeros_like(last_gae_lam)

        delta = rewards[:, t] + gamma * next_values * next_mask - values[:, t]
        last_gae_lam = delta + gamma * lam * next_mask * last_gae_lam
        advantages[:, t] = last_gae_lam * mask[:, t]

    returns = (advantages + values) * mask
    advantages = whiten_masked(advantages, mask) * mask
    return advantages, returns

class PPOTrainer:
    def __init__(self, actor:ActorModel, critic:CriticModel, ref_model, reward_model, config : PPOTrainConfig):
        self.actor = actor
        self.critic = critic
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.config = config

        self.actor_optimizer = AdamW(self.actor.parameters(), lr = config.actor_lr, weight_decay=config.actor_weight_decay)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr = config.critic_lr, weight_decay=config.critic_weight_decay)

    # ===== Codex added: support both raw HF models and wrapped actor-like models =====
    def _forward_logits(self, model, input_ids, attention_mask):
        outputs = model(input_ids, attention_mask=attention_mask)
        return outputs.logits if hasattr(outputs, "logits") else outputs

    # ===== Codex added: helpers for rollout -> buffer writing =====
    def _left_pad_sequences(self, sequences, pad_token_id):
        """
        Codex added:
        Left-pad prompt sequences so the real query tokens stay adjacent to the
        generated response after concatenation.
        """
        max_len = max(x.size(0) for x in sequences)
        batch_size = len(sequences)
        padded = torch.full(
            (batch_size, max_len),
            fill_value=pad_token_id,
            dtype=sequences[0].dtype,
            device=sequences[0].device,
        )

        for i, seq in enumerate(sequences):
            padded[i, max_len - seq.size(0) :] = seq

        return padded

    def _build_full_inputs_from_samples(self, query_tensors, response_tensors, pad_token_id):
        """
        Codex added:
        Rebuild the training-time layout:
        - query block: left padded
        - response block: right padded
        """
        padded_queries = self._left_pad_sequences(query_tensors, pad_token_id)
        padded_responses = pad_sequence(response_tensors, batch_first=True, padding_value=pad_token_id)
        full_input_ids = torch.cat([padded_queries, padded_responses], dim=1)
        attention_mask = full_input_ids.ne(pad_token_id).long()
        query_pad_len = padded_queries.size(1)
        response_lens = torch.tensor(
            [x.size(0) for x in response_tensors],
            dtype=torch.long,
            device=full_input_ids.device,
        )
        return full_input_ids, attention_mask, query_pad_len, response_lens

    def _extract_query_tensors_from_left_padded_batch(self, query_input_ids, query_attention_mask):
        """
        Codex added:
        Remove left padding from the prompt batch and keep only the true query tokens.
        """
        query_tensors = []
        batch_size = query_input_ids.size(0)
        for i in range(batch_size):
            query_tensors.append(query_input_ids[i][query_attention_mask[i].bool()])
        return query_tensors

    def _trim_response_tensor(self, response_tensor, pad_token_id, eos_token_id=None):
        """
        Codex added:
        Remove right padding and optionally stop at the first EOS token.
        """
        response_tensor = response_tensor[response_tensor.ne(pad_token_id)]
        if eos_token_id is not None:
            eos_positions = (response_tensor == eos_token_id).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                first_eos = int(eos_positions[0].item())
                response_tensor = response_tensor[: first_eos + 1]
        return response_tensor

    def _extract_response_tensors_from_generation(
        self,
        query_input_ids,
        generated_ids,
        pad_token_id,
        eos_token_id=None,
    ):
        """
        Codex added:
        HF generate() returns [query_block, generated_block]. We slice off the prompt
        block by batch width, then trim each response sample individually.
        """
        prompt_block_len = query_input_ids.size(1)
        generated_response_block = generated_ids[:, prompt_block_len:]
        response_tensors = []

        batch_size = generated_response_block.size(0)
        for i in range(batch_size):
            response_i = self._trim_response_tensor(
                generated_response_block[i],
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            response_tensors.append(response_i)

        return response_tensors

    def _prepare_rewards_for_buffer(self, rewards, batch_size, device):
        """
        Codex added:
        Normalize rewards into shape [batch].
        """
        if rewards is None:
            raise ValueError(
                "rewards is None. Pass reward scores explicitly, or extend this helper "
                "to match your reward_model interface."
            )

        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        rewards = rewards.to(device=device, dtype=torch.float32).view(-1)
        if rewards.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} rewards, but got {rewards.numel()}.")

        return rewards

    def add_batch_to_buffer_v2(
        self,
        buffer,
        query_input_ids,
        query_attention_mask,
        generated_ids,
        rewards,
        pad_token_id,
        eos_token_id=None,
    ):
        """
        Codex added:
        Write one generated rollout batch into ExperienceBuffer.

        Expected inputs:
        - query_input_ids/query_attention_mask: the left-padded prompt batch used for generate()
        - generated_ids: output of actor.generate(...)
        - rewards: reward score for each sample, shape [batch]

        Stored format in buffer:
        - query_i: unpadded 1D prompt tensor
        - response_i: unpadded 1D response tensor
        - old_logprob_i: 1D response-token logprob tensor
        - old_value_i: 1D response-token value tensor
        - reward_i: scalar tensor
        """
        query_tensors = self._extract_query_tensors_from_left_padded_batch(
            query_input_ids,
            query_attention_mask,
        )
        response_tensors = self._extract_response_tensors_from_generation(
            query_input_ids,
            generated_ids,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        batch_size = len(query_tensors)
        rewards = self._prepare_rewards_for_buffer(rewards, batch_size, query_input_ids.device)

        if any(x.numel() == 0 for x in response_tensors):
            raise ValueError(
                "At least one generated response is empty after trimming. "
                "Check your generation config and EOS/pad handling."
            )

        full_input_ids, attention_mask, query_pad_len, response_lens = self._build_full_inputs_from_samples(
            query_tensors,
            response_tensors,
            pad_token_id,
        )

        with torch.no_grad():
            actor_logits = self.actor(full_input_ids, attention_mask=attention_mask)
            old_log_probs = self.actor.get_response_log_probs_from_full_inputs_v2(
                actor_logits,
                full_input_ids,
                query_pad_len,
                response_lens,
            )

            critic_outputs = self.critic(full_input_ids, attention_mask=attention_mask)
            old_values = self.critic.get_response_values_from_full_values_v2(
                critic_outputs,
                query_pad_len,
                response_lens,
            )


        for i, response_len in enumerate(response_lens.tolist()):
            buffer.add(
                query=query_tensors[i].detach().cpu(),
                response=response_tensors[i].detach().cpu(),
                logprob=old_log_probs[i, :response_len].detach().cpu(),
                value=old_values[i, :response_len].detach().cpu(),
                reward=rewards[i].detach().cpu(),
            )

        response_token_mask = pad_sequence(
            [torch.ones(length, dtype=torch.float32, device=full_input_ids.device) for length in response_lens.tolist()],
            batch_first=True,
            padding_value=0.0,
        )
        with torch.no_grad():
            ref_logits = self._forward_logits(self.ref_model, full_input_ids, attention_mask)
            ref_log_probs = self.actor.get_response_log_probs_from_full_inputs_v2(
                ref_logits,
                full_input_ids,
                query_pad_len,
                response_lens,
            )

            kl_rewards = get_kl_penalty(
                old_log_probs,
                ref_log_probs,
                kl_coef=self.config.kl_coef,
            )
            token_rewards = build_token_level_rewards(
                rewards,
                kl_rewards,
                response_token_mask,
            )
            advantages, returns = compute_advantages_v2(
                old_values,
                token_rewards,
                response_token_mask,
                gamma=self.config.gamma,
                lam=self.config.lam,
            )

        # ===== Codex added: store advantages/returns per sample so buffer stays aligned =====
        for i, response_len in enumerate(response_lens.tolist()):
            buffer.add_adv_ret(
                advantages[i, :response_len].detach().cpu(),
                returns[i, :response_len].detach().cpu(),
                ref_log_probs[i, :response_len].detach().cpu()
            )
        return {
            "num_added": batch_size,
            "query_pad_len": int(query_pad_len),
            "mean_response_len": response_lens.float().mean().item(),
        }
    

    # ===== Codex added: masked losses for padded response-token tensors =====
    def ppo_loss_v2(self, log_probs, old_log_probs, advantages, mask):
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.eps, 1.0 + self.config.eps) * advantages
        clip_mask = (ratio < 1.0 - self.config.eps) | (ratio > 1.0 + self.config.eps)
        clip_frac = clip_mask.float().mean().item()
        loss = -masked_mean(torch.min(surr1, surr2), mask)
        return loss, clip_frac

    def critic_loss_v2(self, returns, values, mask):
        squared_error = (values - returns) ** 2
        return self.config.critic_loss_param * masked_mean(squared_error, mask)

    def train_step_v2(self, buffer_data):
        """
        Codex added:
        A complete PPO train step for the following pipeline:
        1. buffer stores unpadded single-sample query/response tensors
        2. ExperienceBuffer.get_padded_batch pads them into a batch
        3. actor/critic run on concatenated [query + response] inputs
        """
        full_input_ids = buffer_data["full_input_ids"]
        attention_mask = buffer_data["attention_mask"]
        old_log_probs = buffer_data["logprobs"]
        old_values = buffer_data["values"]
        rewards = buffer_data["rewards"]
        query_lens = buffer_data["query_lens"]
        query_pad_len = buffer_data["query_pad_len"]
        response_lens = buffer_data["response_lens"]
        response_token_mask = buffer_data["response_token_mask"].float()
        advantages = buffer_data["advantages"]
        returns = buffer_data["returns"]
        ref_log_prob = buffer_data["ref_log_prob"]
        all_actor_losses = []
        all_critic_losses = []

        total_samples = full_input_ids.size(0)

        for epoch in tqdm.tqdm(range(self.config.ppo_epochs), desc = "PPO buffer epochs"):

            indices = torch.randperm(total_samples)
            
            for start_idx in range(0, total_samples, self.config.mini_batch_size):
                end_idx = start_idx + self.config.mini_batch_size

                mb_indices = indices[start_idx:end_idx]
                mb_response_lens = buffer_data["response_lens"][mb_indices]## 因为这里response_lens是在buffer内部的所有数据的最大response_len， 而不一定等于mini_batch内部的最大response_len
                mb_max_response_len = int(mb_response_lens.max().item())

                mb_full_input_ids = buffer_data["full_input_ids"][mb_indices]
                mb_attention_mask = buffer_data["attention_mask"][mb_indices]
                mb_old_log_probs = buffer_data["logprobs"][mb_indices, :mb_max_response_len]
                mb_advantages = buffer_data["advantages"][mb_indices, :mb_max_response_len]
                mb_returns = buffer_data["returns"][mb_indices ,:mb_max_response_len]
                mb_response_token_mask = buffer_data["response_token_mask"][mb_indices, :mb_max_response_len].float()
                mb_response_lens = buffer_data["response_lens"][mb_indices]
                mb_ref_log_prob = buffer_data["ref_log_prob"][mb_indices, :mb_max_response_len]

                

                # --- 下面进入你原有的训练逻辑 ---
                # Actor Forward
                actor_logits = self.actor(mb_full_input_ids, attention_mask=mb_attention_mask)
                new_log_probs = self.actor.get_response_log_probs_from_full_inputs_v2(
                    actor_logits, mb_full_input_ids, query_pad_len, mb_response_lens
                )

                # Critic Forward
                critic_outputs = self.critic(mb_full_input_ids, attention_mask=mb_attention_mask)
                new_values = self.critic.get_response_values_from_full_values_v2(
                    critic_outputs, query_pad_len, mb_response_lens
                )

                # Loss & Backward
                actor_loss, clip_frac = self.ppo_loss_v2(
                    new_log_probs, mb_old_log_probs, mb_advantages, mb_response_token_mask
                )
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                critic_loss = self.critic_loss_v2(
                    mb_returns, new_values, mb_response_token_mask
                )
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # 记录指标
                all_actor_losses.append(actor_loss.item())
                all_critic_losses.append(critic_loss.item())
                
                # 注意：在 Mini-batch 内部计算真正的监控 KL
                with torch.no_grad():
                    # 这里的 approx_kl 衡量的是当前策略偏离 ref 的程度
                    kl_ref = get_kl_penalty(new_log_probs, mb_ref_log_prob)
                    kl_ref_mean = masked_mean(kl_ref, mb_response_token_mask)

                if self.config.use_wandb:
                    wandb.log({
                        "ppo/actor_loss": actor_loss.item(),
                        "ppo/critic_loss": critic_loss.item(),
                        "ppo/approx_kl": kl_ref_mean.item(),
                        "ppo/clip_fraction": clip_frac,
                        "train/epoch": epoch
                    })

        return {
            "actor_loss": sum(all_actor_losses) / len(all_actor_losses),
            "critic_loss": sum(all_critic_losses) / len(all_critic_losses),
            "mean_reward": rewards.mean().item(), # Reward 对这个 Buffer 是静态的，这样存没问题
        }
