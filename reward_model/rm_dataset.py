from torch.utils.data import Dataset
import json
from pathlib import Path


class RewardDataset(Dataset):
    def __init__(self, path, strip_text = True):
        super().__init__()
        self.prompt_keys = ["instruction", "prompt", "input", "question"]
        self.path = Path(path)
        datas = self._load_file()
        self.strip_text = strip_text
        self.data = []
        
        for data in datas:
            if any(k in data for k in self.prompt_keys) and all(k in data for k in ["chosen", "rejected"]):
                self.data.append(data)
        print(f"成功加载数据集: {len(self.data)} 条有效样本 (原始数据: {len(datas)} 条)")


    def _load_file(self):
        datas = []
        if not self.path.exists():
            raise ValueError(f"the path {self.path} of the json data doesn't exist")

        with open(str(self.path), "r", encoding="utf-8") as f:
            if self.path.suffix == ".jsonl":
                for line in f :
                    line = line.strip()
                    if line:
                        datas.append(json.loads(line))
            elif self.path.suffix == ".json" :
                datas = json.load(f)
            else :
                raise ValueError("only support .json or .jsonl format")
        return datas
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        raw_item = self.data[index]
        prompt = ""
        for key in self.prompt_keys:
            if key in raw_item :
                prompt = raw_item[key]
                break
        chosen = raw_item.get("chosen", "")
        rejected = raw_item.get("rejected", "")
        if self.strip_text:
            prompt = prompt.strip()
            chosen = chosen.strip()
            rejected = rejected.strip()
        return {
            "prompt" : prompt,
            "chosen" : chosen,
            "rejected" : rejected
        }
    
class UltraRewardDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]

        prompt = item["prompt"]
        chosen_msgs = item["chosen"]
        rejected_msgs = item["rejected"]

        chosen_content = chosen_msgs[-1]["content"]
        rejected_content = rejected_msgs[-1]["content"]


        return {
            "prompt" : prompt,
            "chosen": chosen_content,
            "rejected": rejected_content
        }

