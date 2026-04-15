from dataclasses import dataclass, field
@dataclass
class PPOTrainConfig:
    model_name: str
    train_file: str
    eval_file: str
    output_dir: str
    dataset_name : str
    prompt_key :  str
    reward_model_name: str
    reward_checkpoint_path: str

    rollout_batch_size: int
    mini_batch_size: int
    ppo_epochs: int
    total_episodes: int
    buffer_target_size: int

    max_prompt_length: int
    max_new_tokens: int
    max_total_length: int

    actor_lr: float
    critic_lr: float
    actor_weight_decay: float
    critic_weight_decay: float

    gamma: float
    lam: float
    eps: float
    kl_coef: float
    critic_loss_param: float

    pad_token_id: int | None = None
    eos_token_id: int | None = None

    use_wandb: bool = False
    seed: int = 42
    device: str = "cuda"
    actor_lora_r: int = 8
    actor_lora_alpha: int = 16
    actor_lora_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    critic_lora_r: int = 8
    critic_lora_alpha: int = 16
    critic_lora_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    eval_every: int = 5
    eval_batch_size: int = 4
    eval_max_new_tokens: int = 128
    num_eval_batches: int = 10


    # ===== Codex added: derived LoRA configs passed directly into model builders =====
    actor_lora_config: dict = field(init=False)
    critic_lora_config: dict = field(init=False)

    def __post_init__(self):
        self.actor_lora_config = {
            "r": self.actor_lora_r,
            "alpha": self.actor_lora_alpha,
            "modules": self.actor_lora_modules,
        }
        self.critic_lora_config = {
            "r": self.critic_lora_r,
            "alpha": self.critic_lora_alpha,
            "modules": self.critic_lora_modules,
        }
