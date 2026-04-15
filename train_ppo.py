import argparse
import torch
from utils.data_utils import build_dataloader
from trainer.build import build_models, build_trainer_and_buffer, build_tokenizer
import random
from utils.rollout import collect_rollout_buffer,ppo_update, save_checkpoint
from utils.config import PPOTrainConfig
from utils.evaluate_model import evaluate
import wandb

def build_parser():
    parser = argparse.ArgumentParser(description="the configuration of the PPO train")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type =str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--reward_model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--reward_checkpoint_path", type=str, default="reward_model/results/best/reward_adapter.pth")

    parser.add_argument("--rollout_batch_size", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--total_episodes", type=int, default=100)
    parser.add_argument("--buffer_target_size", type=int, default=32)
    parser.add_argument("--num_eval_batches", type = int, default=8)
    parser.add_argument("--eval_batch_size", type = int, default=4)
    parser.add_argument("--eval_every", type = int, default=5)

    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_total_length", type=int, default= 512)
    parser.add_argument("--eval_max_new_tokens", type= int, default = 256)

    parser.add_argument("--actor_lr", type=float, default=1e-5)
    parser.add_argument("--critic_lr", type=float, default=1e-5)
    parser.add_argument("--actor_weight_decay", type=float, default=0.0)
    parser.add_argument("--critic_weight_decay", type=float, default=0.0)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.01)
    parser.add_argument("--critic_loss_param", type=float, default=1.0)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--actor_lora_r", type=int, default=8)
    parser.add_argument("--actor_lora_alpha", type=int, default=16)
    parser.add_argument("--actor_lora_modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--critic_lora_r", type=int, default=8)
    parser.add_argument("--critic_lora_alpha", type=int, default=16)
    parser.add_argument("--critic_lora_modules", nargs="+", default=["q_proj", "v_proj"])
    return parser

def load_config(args)-> PPOTrainConfig:
    return PPOTrainConfig(
        model_name=args.model_name,
        train_file=args.train_file,
        eval_file = args.eval_file,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        prompt_key=args.prompt_key,
        reward_model_name=args.reward_model_name,
        reward_checkpoint_path=args.reward_checkpoint_path,
        rollout_batch_size=args.rollout_batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        total_episodes=args.total_episodes,
        buffer_target_size=args.buffer_target_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        max_total_length=args.max_total_length,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        actor_weight_decay=args.actor_weight_decay,
        critic_weight_decay=args.critic_weight_decay,
        gamma=args.gamma,
        lam=args.lam,
        eps=args.eps,
        kl_coef=args.kl_coef,
        critic_loss_param=args.critic_loss_param,
        use_wandb=args.use_wandb,
        seed=args.seed,
        device=args.device,
        actor_lora_r=args.actor_lora_r,
        actor_lora_alpha=args.actor_lora_alpha,
        actor_lora_modules=args.actor_lora_modules,
        critic_lora_r=args.critic_lora_r,
        critic_lora_alpha=args.critic_lora_alpha,
        critic_lora_modules=args.critic_lora_modules,
        eval_every = args.eval_every,
        eval_batch_size = args.eval_batch_size,
        eval_max_new_tokens = args.eval_max_new_tokens,
        num_eval_batches = args.num_eval_batches

    )
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args)
    set_seed(config.seed)

    ##build tokenizer
    actor_tokenizer = build_tokenizer(config)
    ## build data
    train_dataloader, eval_dataloader = build_dataloader(config, actor_tokenizer)
    actor, critic, ref_model, reward_model, reward_tokenizer = build_models(config)
    trainer, buffer = build_trainer_and_buffer(config, actor, critic, ref_model, reward_model)

    train_iter = iter(train_dataloader)
    for episode in range(config.total_episodes):
        train_iter = collect_rollout_buffer(train_iter, train_dataloader, trainer, buffer, actor_tokenizer, reward_tokenizer, config)
        train_stats = ppo_update(trainer, buffer, config)
        print({
            "episode" : episode,
            **train_stats,
            "buffer_size" : len(buffer.queries)
        })

        if (episode + 1) % 10 == 0:
            save_checkpoint(trainer, actor_tokenizer, config, episode+1)
        
        if (episode + 1) % config.eval_every == 0:
            eval_metrics, samples = evaluate(
                trainer=trainer,
                eval_dataloader=eval_dataloader,
                actor_tokenizer=actor_tokenizer,
                reward_tokenizer=reward_tokenizer,
                config=config
            )
            
            if config.use_wandb:
                wandb.log({
                    "eval/reward_mean": eval_metrics["eval/reward_mean"],
                    "eval/reward_std": eval_metrics["eval/reward_std"],
                    "eval/response_len_mean": eval_metrics["eval/response_len_mean"],
                    "eval/eos_rate": eval_metrics["eval/eos_rate"],
                    "eval/empty_response_rate": eval_metrics["eval/empty_response_rate"]
                })

            for sample in samples[:3]:
                print("PROMPT:", sample["prompt"])
                print("RESPONSE:", sample["response"])
                print("REWARD:", sample["reward"])
                print("-" * 40)
                

        buffer.clear()

if __name__ == "__main__":
    main()


            
        





