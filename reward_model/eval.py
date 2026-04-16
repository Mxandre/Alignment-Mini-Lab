import argparse
from transformers import AutoTokenizer
from reward_model.model import MyRewardModel
import torch
from reward_model.rm_dataset import RewardDataset, UltraRewardDataset
from reward_model.collator import MyRewardCollator
from torch.utils.data import DataLoader
import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description = "Evaluate the Reward Model")
    parser.add_argument("--reward_model_name", type = str, default = "Qwen/Qwen2.5-7B", help = "The Model name")
    parser.add_argument("--reward_checkpoint_path", type = str, default = "reward_model_v1/reward_adapter.pth", help = "Save_model_path" )
    parser.add_argument("--test_path", type = str, default= "data/test_rm.jsonl", help = "test data path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_csv", type = str, default = "reward_model/results/test.csv", help = "results save path")
    parser.add_argument("--save_fig_path", type = str, default = "reward_model/results/fig", help = "Path to save the figure")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    print(f"Loading the checkpoint from {args.reward_checkpoint_path}")
    checkpoint = torch.load(args.reward_checkpoint_path, map_location = "cpu")
    saved_args = checkpoint["args"]
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)

    model = MyRewardModel(args.reward_model_name, saved_args['lora_r'], saved_args['lora_alpha'], saved_args['lora_dropout'])

    
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if any("value_head" in k for k in state_dict.keys()):
        print("value head can be found")
    else :
        print("cannot find the value head")

    model.load_state_dict(state_dict, strict = False)

    model.to("cuda")
    model.eval()
    return model, tokenizer

def main():
    args = parse_args()

    # test_dataset = RewardDataset(args.test_path)  here you can use your own dataset
    data_path = "/root/autodl-tmp/datasets/ultrafeedback_binarized"

    raw_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", cache_dir = data_path)
    test_dataset = UltraRewardDataset(raw_ds["test"])

    model, tokenizer = load_model_and_tokenizer(args)
    test_collator = MyRewardCollator(tokenizer, max_length=args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size= args.batch_size, shuffle = False, collate_fn=test_collator)

    print("starting evaluation")
    results = []
    total_count = 0
    correct_count = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader):
            c_ids, c_mask = batch["chosen_input_ids"].to("cuda"), batch["chosen_attention_mask"].to("cuda")
            r_ids, r_mask = batch["rejected_input_ids"].to("cuda"), batch["rejected_attention_mask"].to("cuda")

            chosen_reward  = model(c_ids, c_mask)
            rejected_reward = model(r_ids, r_mask)
            batch_size = c_ids.size(0)
            total_count += batch_size
            correct_count += (chosen_reward > rejected_reward).sum()
            c_list = chosen_reward.cpu().float().tolist()
            r_list = rejected_reward.cpu().float().tolist()
            
            for c_score, r_score in zip(c_list, r_list):
                results.append({
                    "chosen_score": c_score,
                    "rejected_score": r_score,
                    "margin": c_score - r_score,
                    "correct": c_score > r_score
                })
    final_acc = correct_count / total_count
    print(f"\n" + "="*30)
    print(f"Test Accuracy: {final_acc:.4f}")
    print(f"="*30)

    df = pd.DataFrame(results)
    save_fig_path = Path(args.save_fig_path)
    save_fig_path.mkdir(parents = True, exist_ok=True)
    # --- 可视化 1: 奖励得分分布图 (Double Density Plot) ---
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="chosen_score", fill=True, label="Chosen", color="green")
    sns.kdeplot(data=df, x="rejected_score", fill=True, label="Rejected", color="red")
    plt.title("Reward Score Distribution on Test Set")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"{args.save_fig_path}/reward_distribution.png")
    plt.show()

    # --- 可视化 2: Margin 的直方图 ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df['margin'], bins=30, kde=True, color="blue")
    plt.axvline(x=0, color='red', linestyle='--') # 0刻度线，左边是错的，右边是对的
    plt.title("Reward Margin Distribution (Chosen - Rejected)")
    plt.savefig(f"{args.save_fig_path}/margin_distribution.png")
    plt.show()
    df.to_csv(args.output_csv, index=False)

if __name__ == "__main__" : 
    main()

