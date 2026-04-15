import torch
from reward_model.rm_dataset import RewardDataset, UltraRewardDataset
from reward_model.collator import MyRewardCollator
from torch.utils.data import DataLoader
from reward_model.model import MyRewardModel
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import argparse
import tqdm
from reward_model.loss import log_sigmoid_loss
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
from reward_model.utils import EarlyStopping
import wandb
from datasets import load_dataset

WAND_KEY = "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"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Reward Model")

    parser.add_argument("--train_path", type = str, default = "data/dpo_train.jsonl",  help = "Path to training data")
    parser.add_argument("--eval_path", type=str, default = "data/dpo_eval.jsonl", help="Path to eval data")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B", help="Base model name or path")
    parser.add_argument("--save_path", type = str, default = "reward_model/results/best", help = "Path to save the model")

    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--weight_decay", type=float, default= 0.0)
    parser.add_argument("--lora_r", type = int, default= 16)
    parser.add_argument("--lora_alpha", type = int, default=32)
    parser.add_argument("--seed", type = int, default= 42)

    parser.add_argument("--patience", type=int, default=3, help="How many epochs to wait before stopping")
    return parser.parse_args()


def main():
    args = parse_args()
    if args is None:
        print("错误：args 为空，请检查 parse_args 函数是否正确 return")
        return
    
    wandb.init(
    project="mini-reward-model", 
    config=vars(args),  # 自动记录你的命令行参数
    name=f"run-{args.model_name.split('/')[-1]}-{args.lr}" # 自动命名实验
)
    
    # 1. 加载 Tokenizer (建议从模型路径加载)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 2. 准备数据 (传入 args 里的路径) here , is your own dataset config
    # train_dataset = RewardDataset(args.train_path)
    # eval_dataset = RewardDataset(args.eval_path)

    ## use ultra ndataset

    raw_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    split_results = raw_ds["train_prefs"].train_test_split(test_size = 0.2, seed = args.seed )
    train_dataset = UltraRewardDataset(split_results["train"])
    eval_dataset = UltraRewardDataset(split_results["test"])
    collator = MyRewardCollator(tokenizer, max_length=args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle = False, collate_fn = collator)
    
    # 3. 初始化模型和调度器
    model = MyRewardModel(args.model_name).to("cuda")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience= args.patience)
    
    total_steps = len(train_dataloader) * args.num_epochs
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # 4. 训练循环
    best_acc = 0.0
    checkpoint = {}
    trainable_params_names = {n for n, p in model.named_parameters() if p.requires_grad}
    accumulation_steps = args.accumulation_steps
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            c_ids, c_mask = batch["chosen_input_ids"].to("cuda"), batch["chosen_attention_mask"].to("cuda")
            r_ids, r_mask = batch["rejected_input_ids"].to("cuda"), batch["rejected_attention_mask"].to("cuda")
            with torch.amp.autocast(dtype = torch.bfloat16):
                r_chosen = model(c_ids, c_mask)
                r_rejected = model(r_ids, r_mask)
                loss = log_sigmoid_loss(r_chosen, r_rejected)
                loss = loss / accumulation_steps

            loss.backward()
            if (i+1) % accumulation_steps : 
                grad_norm = clip_grad_norm_(model.parameters(), max_norm = 1)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            total_loss += loss.item() * accumulation_steps
            wandb.log({
            "train/loss": loss.item(),
            "train/lr": lr_scheduler.get_last_lr()[0],
            "train/margin": (r_chosen - r_rejected).mean().item(),
            "train/grad_norm": grad_norm.item()
        })
        print(f"Epoch {epoch} Average Loss : {total_loss / len(train_dataloader)}")

        with torch.no_grad():
            accuracy_list = []
            all_chosen_rewards = []
            all_rejected_rewards = []
            model.eval()
            for batch in tqdm.tqdm(eval_dataloader, desc = f"Eval Epoch{epoch}"):
                c_ids, c_mask = batch["chosen_input_ids"].to("cuda"), batch["chosen_attention_mask"].to("cuda")
                r_ids, r_mask = batch["rejected_input_ids"].to("cuda"), batch["rejected_attention_mask"].to("cuda")
                
                r_chosen = model(c_ids, c_mask)
                r_rejected = model(r_ids, r_mask)
                batch_size = c_ids.size(0)
                accuracy_list.append((r_chosen > r_rejected).sum() / batch_size)

                all_chosen_rewards.extend(r_chosen.cpu().float().tolist())
                all_rejected_rewards.extend(r_rejected.cpu().float().tolist())
            all_accs = torch.stack(accuracy_list)
            accuracy = all_accs.mean().item()
            wandb.log({
            "val/accuracy": accuracy,
            "val/best_accuracy": best_acc if accuracy <= best_acc else accuracy,
            "val/chosen_rewards_dist": wandb.Histogram(all_chosen_rewards),
            "val/rejected_rewards_dist": wandb.Histogram(all_rejected_rewards),
        })

            if accuracy > best_acc:
                best_acc = accuracy
                save_dir = Path(args.save_path)
                save_dir.mkdir(parents=True, exist_ok=True) 
                trainable_state_dict = {
                    k:v for k,v in model.state_dict().items() if k in trainable_params_names
                }
                checkpoint = {
                    'model_state_dict': trainable_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'args': vars(args), # 保存当时的命令行参数，方便以后查阅
                }
                torch.save(checkpoint, f"{args.save_path}/reward_adapter.pth")
            print(f"val_accuracy : {accuracy}")
            early_stopping(accuracy)
            if early_stopping.early_stop:
                print("Early stopping triggered. Training finished.")
                break # 跳出 Epoch 循环

if __name__ == "__main__":
    main()