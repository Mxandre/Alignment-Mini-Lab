import json
import argparse
from reward_model.eval import load_model_and_tokenizer
from reward_model.collator import SimpleInferenceCollator
from torch.utils.data import DataLoader
import torch
import tqdm
from pathlib import Path
import pandas as pd
def build_parse():
    parser = argparse.ArgumentParser(description="compare the reward model with gpt judge")
    parser.add_argument(
        "--beta",
        default=0.3,
        help = "the beta used in DPO"
    )
    parser.add_argument("--model_name", type = str, default = "Qwen/Qwen2.5-1.5B", help = "The Model name")
    parser.add_argument("--checkpoint_path", type = str, default = "reward_model/results/best/reward_adapter.pth", help = "Save_model_path" )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="reward_model/results/analysis", help="Dir to save CSV")
    return parser


def load_json_results(path_dataset, path_gpt_judge):
    path_dataset = Path(path_dataset)
    path_gpt_judge = Path(path_gpt_judge)
    dataset_A = []
    dataset_B = []
    truth = []
    gpt_judge = []
    with open(path_dataset, "r", encoding="utf-8") as f:
        if path_dataset.suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if line :
                    data = json.loads(line)
                    p = data["instruction"]
                    response_a = data["answer_a"]
                    response_b = data["answer_b"]
                    dataset_A.append(f"Prompt: {p}\n\nResponse: {response_a}")
                    dataset_B.append(f"Prompt: {p}\n\nResponse: {response_b}")
                    truth.append(data["truth"])
        elif path_dataset.suffix == ".json":
            lines = json.load(f)
            for line in lines:
                line = line.strip()
                if line :
                    data = json.loads(line)
                    p = data["instruction"]
                    response_a = data["answer_a"]
                    response_b = data["answer_b"]
                    dataset_A.append(f"Prompt: {p}\n\nResponse: {response_a}")
                    dataset_B.append(f"Prompt: {p}\n\nResponse: {response_b}")
                    truth.append(data["truth"])
        else :
            raise ValueError("only support .json or .jsonl format")
        
    with open(path_gpt_judge, "r", encoding = "utf-8") as f:
        if path_gpt_judge.suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if line :
                    data = json.loads(line)
                    gpt_judge.append(data["winner"])
    print(f"Loading the {len(dataset_A)} data")
    return dataset_A, dataset_B, truth, gpt_judge

def main():
    parser = build_parse()
    args = parser.parse_args()
    beta_str = str(args.beta).replace('.', '_')
    path_dataset = f"outputs/eval/results_beta_{beta_str}.jsonl"  ## x
    path_gpt_judge = f"outputs/eval/final_evaluation_results_beta_{beta_str}.jsonl"
    dataset_A, dataset_B, truth, gpt_judge = load_json_results(path_dataset, path_gpt_judge)
    model, tokenizer = load_model_and_tokenizer(args)
    collator = SimpleInferenceCollator(tokenizer, args.max_length)
    
    loader_A = DataLoader(dataset_A, batch_size=args.batch_size, shuffle = False, collate_fn= collator)
    loader_B = DataLoader(dataset_B, batch_size=args.batch_size, shuffle= False, collate_fn=collator)

    def get_scores(loader):
        scores = []
        with torch.no_grad():
            for batch in tqdm.tqdm(loader, desc = "Predicting Scores"):
                input_ids = batch["input_ids"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")

                rewards = model(input_ids, attention_mask)
                scores.extend(rewards.cpu().float().tolist())
        return scores
    
    score_A = get_scores(loader_A)
    score_B = get_scores(loader_B)
    results = []
    for i in range(len(score_A)):
        model_a_name = truth[i] 
        model_b_name = "SFT" if model_a_name == "DPO" else "DPO"

        if score_A[i] > score_B[i] :
            rm_winner = model_a_name
        else:
            rm_winner = model_b_name
        
        is_match = (rm_winner == gpt_judge[i])

        results.append({
            "prompt_idx" : i,
            "rm_score_a" : score_A[i],
            "rm_score_B" : score_B[i],
            "margin" : score_A[i] - score_B[i],
            "abs_margin" : abs(score_A[i] - score_B[i]),
            "rm_winner" : rm_winner,
            "gpt_winner" : gpt_judge[i],
            "is_match" : is_match
        })
    
        
    df = pd.DataFrame(results)
    Path(args.output_dir).mkdir(parents = True, exist_ok=True)
    csv_filename = f"{args.output_dir}/rm_vs_gpt_beta_{beta_str}.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    print(f"\n✅ 详细分析结果已保存至: {csv_filename}")

    valid_df = df[df["gpt_winner"] != "Tie"].copy()
    if not valid_df.empty:
        overall_acc = valid_df["is_match"].mean()
        print(f"📊 总一致率 (排除 Tie): {overall_acc:.4f}")

        # --- 置信度分层分析 ---
        # 将 abs_margin 分成四个分位数区间
        valid_df['confidence_bin'] = pd.qcut(valid_df['abs_margin'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        bin_stats = valid_df.groupby('confidence_bin', observed=True)['is_match'].mean()
        
        print("\n📈 置信度（Margin）与 GPT 一致率的关系:")
        for bin_name, acc in bin_stats.items():
            print(f"  [{bin_name} Confidence] Agreement: {acc:.4f}")
    else:
        print("⚠ 警告：没有可用的非 Tie 样本进行统计。")
                
if __name__ == "__main__":
    main()
                

    
