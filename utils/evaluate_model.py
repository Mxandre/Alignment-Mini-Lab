
from trainer.ppo_trainer import PPOTrainer
from utils.rollout import compute_rewards
import torch
import numpy as np

def decode_eval_responses(
    input_ids,
    generated_ids,
    tokenizer,
    eos_token_id,
    pad_token_id,
):
    prompt_len = input_ids.size(1)
    response_ids = generated_ids[:, prompt_len:]

    responses = []
    response_lengths = []
    eos_flags = []

    for row in response_ids:
        row = row[row.ne(pad_token_id)]

        eos_found = False
        if eos_token_id is not None:
            eos_pos = (row == eos_token_id).nonzero(as_tuple=False)
            if eos_pos.numel() > 0:
                row = row[: int(eos_pos[0].item()) + 1] ## 我们需要token长度， 而不是decode出来的字符长度
                eos_found = True

        text = tokenizer.decode(row, skip_special_tokens=True)
        responses.append(text)
        response_lengths.append(int(row.numel()))
        eos_flags.append(int(eos_found))

    return responses, response_lengths, eos_flags

def summarize_eval_metrics(rewards, response_lengths, eos_flags):
    return{
    "eval/reward_mean": float(np.mean(rewards)) if rewards else 0.0,
    "eval/reward_std": float(np.std(rewards)) if rewards else 0.0,
    "eval/response_len_mean": float(np.mean(response_lengths)) if response_lengths else 0.0,
    "eval/eos_rate": float(np.mean(eos_flags)) if eos_flags else 0.0,
    "eval/empty_response_rate": float(np.mean([x == 0 for x in response_lengths])) if response_lengths else 0.0,
    }

def evaluate_one_batch(
    batch,
    trainer : PPOTrainer,
    actor_tokenizer,
    reward_tokenizer,
    config
):
    
    input_ids = batch["input_ids"].to(config.device)
    attention_mask = batch["attention_mask"].to(config.device)
    prompts = batch["prompts"]
    
    generated_ids = trainer.actor.generate(
        input_ids = input_ids,
        attention_mask = attention_mask,
        max_new_tokens = config.eval_max_new_tokens,
        do_sample = False,
        pad_token_id = config.pad_token_id,
        eos_token_id = config.eos_token_id
    )

    rewards = compute_rewards(
        input_ids=input_ids,
        generated_ids=generated_ids,
        actor_tokenizer=actor_tokenizer,
        reward_tokenizer=reward_tokenizer,
        reward_model=trainer.reward_model,
        config=config,
    )

    responses, response_lengths, eos_flags = decode_eval_responses(
        input_ids=input_ids,
        generated_ids=generated_ids,
        tokenizer=actor_tokenizer,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
    )

    batch_metrics = {
        "rewards": rewards.detach().cpu().tolist(),
        "response_lengths": response_lengths,
        "eos_flags": eos_flags,
    }

    batch_samples = []
    for prompt, response, reward in zip(prompts, responses, rewards.detach().cpu().tolist()):
        batch_samples.append({
            "prompt": prompt,
            "response": response,
            "reward": reward,
        })

    return batch_metrics, batch_samples

def evaluate(
        trainer,
        eval_dataloader,
        actor_tokenizer,
        reward_tokenizer,
        config
        ):
    trainer.actor.eval()
    all_rewards = []
    all_lengths = []
    all_eos_flags = []
    sample_records = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if step >= config.num_eval_batches :
                break

            batch_metrics, batch_samples = evaluate_one_batch(
                batch,
                trainer,
                actor_tokenizer,
                reward_tokenizer,
                config
            )
            all_rewards.extend(batch_metrics["rewards"])
            all_lengths.extend(batch_metrics["response_lengths"])
            all_eos_flags.extend(batch_metrics["eos_flags"])

            sample_records.extend(batch_samples)
    eval_metrics = summarize_eval_metrics(all_rewards, all_lengths, all_eos_flags)
    trainer.actor.train()
    return eval_metrics, sample_records

    

