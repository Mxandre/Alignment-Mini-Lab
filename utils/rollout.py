
import torch
from utils.config import PPOTrainConfig
from trainer.ppo_trainer import PPOTrainer
from trainer.buffer import ExperienceBuffer
from pathlib import Path
def compute_rewards(input_ids, generated_ids, actor_tokenizer, reward_tokenizer, reward_model, config):
    
    device = generated_ids.device
    q_len = input_ids.size(1)
    res_ids = generated_ids[:, q_len:]

    # 2. 提取 Query 的 Token IDs
    query_ids = generated_ids[:, :q_len]

    prompts_text = actor_tokenizer.batch_decode(query_ids, skip_special_tokens = True)
    res_text = actor_tokenizer.batch_decode(res_ids, skip_special_tokens = True)


    texts_for_rm = [f"Prompt: {p.strip()}\n\nResponse: {r.strip()} " for p, r in zip(prompts_text, res_text)]
    
    rm_inputs = reward_tokenizer(
        texts_for_rm,
        return_tensors = "pt",
        padding = True,
        truncation = True,
        max_length = config.max_total_length
    ).to(device)

    reward_model.eval()
    with torch.no_grad():
        outputs = reward_model(**rm_inputs)
        if hasattr(outputs, "logits"):
            rewards = outputs.logits
        else:
            rewards = outputs

    return rewards.view(-1).float()

def rollout_one_batch(batch, trainer:PPOTrainer, buffer:ExperienceBuffer, actor_tokenizer, reward_tokenizer, config:PPOTrainConfig):
    input_ids = batch["input_ids"].to(config.device)
    attention_mask = batch["attention_mask"].to(config.device)

    with torch.no_grad():
        generated_ids = trainer.actor.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = config.max_new_tokens,
            do_sample = True,
            top_p = 0.9,
            temperature = 1.0,
            pad_token_id = config.pad_token_id,
            eos_token_id = config.eos_token_id,
        )

        rewards = compute_rewards(
            input_ids = input_ids,
            generated_ids = generated_ids,
            actor_tokenizer = actor_tokenizer,
            reward_tokenizer = reward_tokenizer,
            reward_model = trainer.reward_model,
            config = config
        )

        stats = trainer.add_batch_to_buffer_v2(
            buffer = buffer,
            query_input_ids=input_ids,
            query_attention_mask=attention_mask,
            generated_ids=generated_ids,
            rewards=rewards, 
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id
        )

        return stats
    
def collect_rollout_buffer(data_iter, dataloader, trainer:PPOTrainer, buffer:ExperienceBuffer, actor_tokenizer, reward_tokenizer, config: PPOTrainConfig):
    buffer.clear()

    while(len(buffer.queries) < config.buffer_target_size) : 
        try :
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        rollout_one_batch(batch, trainer, buffer, actor_tokenizer, reward_tokenizer, config)
    return data_iter
    
def ppo_update(trainer:PPOTrainer, buffer:ExperienceBuffer, config : PPOTrainConfig):
    buffer_data = buffer.get_padded_batch(
        pad_token_id= config.pad_token_id,
        device = config.device
    )
    train_stats = trainer.train_step_v2(buffer_data)
    return train_stats

def save_checkpoint(trainer:PPOTrainer, tokenizer, config:PPOTrainConfig, step:int):
    output_dir = Path(f"{config.output_dir}/checkpoint-{step}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 Actor
    torch.save(trainer.actor.state_dict(), output_dir / "actor.pt")
    ## merged_actor = trainer.actor.backbone.merge_and_unload()
    ## merge_actor.save_pretrained(output_dir/"actor_final")

    # 保存 Critic
    torch.save(trainer.critic.state_dict(), output_dir / "critic.pt")
    tokenizer.save_pretrained(output_dir)


