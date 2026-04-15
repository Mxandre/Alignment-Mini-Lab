from models.actor_critic import ActorModel, CriticModel
from utils.config import PPOTrainConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from reward_model.eval import load_model_and_tokenizer
from trainer.ppo_trainer import PPOTrainer
from trainer.buffer import ExperienceBuffer

def build_tokenizer(config: PPOTrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None :
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    return tokenizer

def build_models(config: PPOTrainConfig):
    actor = ActorModel(
        config.model_name,
        config.actor_lora_config
    )
    
    critic = CriticModel(
        config.model_name,
        config.critic_lora_config
    )
    
    ref_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype = torch.bfloat16, device_map = "cuda")
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()
    
    reward_model, reward_tokenizer = load_model_and_tokenizer(config)   ###这里仍然待完善加载逻辑
    return actor, critic, ref_model, reward_model, reward_tokenizer

def build_trainer_and_buffer(config, actor, critic, ref_model, reward_model):
    trainer = PPOTrainer(
        actor=actor,
        critic=critic,
        ref_model=ref_model,
        reward_model=reward_model,
        config = config
    )
    buffer = ExperienceBuffer()
    return trainer, buffer
