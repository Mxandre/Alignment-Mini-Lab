# alignment-mini-lab

A compact alignment lab for comparing a base model, an SFT-adapted model, and a DPO-adapted model on instruction-following behavior.

## Project Goal

This project studies whether supervised fine-tuning (SFT) and direct preference optimization (DPO) improve instruction obedience relative to the same base model.

The comparison is centered on three simple controllable instruction types:

- instruction constraints
- style differences
- format requirements

We use `Qwen/Qwen2.5-1.5B` as the shared base model, train an SFT adapter on synthetic instruction-following data, then continue from the SFT checkpoint with DPO training. Final responses on the held-out test set are judged by `gpt-4o-mini`.

## Dataset Design

The datasets are generated with GPT and organized into three categories:

- `instruction constraints`: prompts that require obeying explicit lexical or content restrictions
- `style differences`: prompts that require a target tone, persona, or style
- `format requirements`: prompts that require a specific structural output format

Current split sizes:

- train: `80 x 3 = 240` examples
- eval: `10 x 3 = 30` examples
- test: `15 x 3 = 45` examples

Files and saved artifacts in `data/`:

- [sft_train.jsonl](data/sft_train.jsonl)
- [sft_eval.jsonl](data/sft_eval.jsonl)
- [dpo_train.jsonl](data/dpo_train.jsonl)
- [dpo_eval.jsonl](data/dpo_eval.jsonl)
- [test.jsonl](data/test.jsonl)
- [sft_dataset](data/sft_dataset)
- [dpo_dataset](data/dpo_dataset)
- [test_dataset](data/test_dataset)

## Training Setup

### Base model

- model: `Qwen/Qwen2.5-1.5B`
- tokenizer: shared tokenizer from the same base model family
- precision: `bfloat16`
- adapter method: LoRA

### SFT stage

The SFT experiment is currently run from [01_sft_minimal.ipynb](notebooks/01_sft_minimal.ipynb) with:

- LoRA rank `r=16`
- `lora_alpha=32`
- `lora_dropout=0.2`
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- batch size per device: `2`
- gradient accumulation: `4`
- epochs: `10`
- learning rate: `2e-5`
- optimizer: `adamw_torch`
- scheduler: `cosine`
- weight decay: `0.05`
- warmup steps: `30`
- eval strategy: `epoch`
- logging strategy: `steps`
- logging steps: `1`
- report target: TensorBoard

Current SFT checkpoints and logs live under:

- [outputs/sft](outputs/sft)

### DPO stage

The DPO experiment is currently run from [02_dpo_minimal.ipynb](notebooks/02_dpo_minimal.ipynb) by loading the SFT checkpoint as the starting PEFT model and training on preference pairs.

Current DPO settings include:

- initialization: SFT checkpoint from `outputs/sft/checkpoint-310`
- batch size per device: `2`
- gradient accumulation: `4`
- epochs: `10`
- learning rate: `5e-7`
- optimizer: `adamw_torch`
- scheduler: `cosine`
- weight decay: `0.05`
- warmup steps: `30`
- eval strategy: `epoch`
- logging strategy: `steps`
- logging steps: `1`
- save strategy: `epoch`
- early stopping patience: `2`
- model selection metric: `eval_loss`

For DPO, we run a beta grid search to study how `beta` affects downstream obedience. The current experiment set includes:

- `beta = 0.2`
- `beta = 0.3`
- `beta = 0.4`
- `beta = 0.5`

DPO outputs and checkpoints live under:

- [outputs/dpo](outputs/dpo)
- beta-specific runs such as `outputs/dpo/dpo_beta_0_2`, `outputs/dpo/dpo_beta_0_3`, `outputs/dpo/dpo_beta_0_4`, and `outputs/dpo/dpo_beta_0_5`

## Evaluation

Evaluation focuses on comparing responses from:

- the base model
- the SFT adapter
- the DPO adapter

The main held-out comparison uses the test split and a GPT-based judge:

- judge model: `gpt-4o-mini` and `Phi-4`

The repository also stores generated evaluation outputs such as:

- [results.jsonl](outputs/results.jsonl)
- [gpt_judge.jsonl](outputs/gpt_judge.jsonl)
- [final_evaluation_results.jsonl](outputs/final_evaluation_results.jsonl)

For training diagnostics, metric plots are generated from `trainer_state.json` and saved as PNG files with [plot_training_metrics.py](src/plot_training_metrics.py).

## Repository Layout

```text
alignment-mini-lab/
├── README.md
├── requirements.txt
├── configs/
│   ├── sft.yaml
│   └── dpo.yaml
├── data/
│   ├── sft_train.jsonl
│   ├── sft_eval.jsonl
│   ├── dpo_train.jsonl
│   ├── dpo_eval.jsonl
│   ├── test.jsonl
│   ├── sft_dataset/
│   ├── dpo_dataset/
│   └── test_dataset/
├── notebooks/
│   ├── 01_sft_minimal.ipynb
│   ├── 02_dpo_minimal.ipynb
│   └── 03_compare_models.ipynb
├── src/
│   ├── build_sft_dataset.py
│   ├── build_dpo_dataset.py
│   ├── plot_training_metrics.py
|
└── outputs/
    ├── sft/
    ├── dpo/
    └── eval/
```

## Current Notes

- The current workflow is notebook-first, with [01_sft_minimal.ipynb](notebooks/01_sft_minimal.ipynb) and [02_dpo_minimal.ipynb](notebooks/02_dpo_minimal.ipynb) serving as the main experiment drivers.
- SFT and DPO builders in `src/` normalize local JSONL files into Hugging Face `DatasetDict` objects.
- DPO training currently explores the effect of `beta` rather than claiming a single best final setting.

## Interesing Discovery
We implement a token-level-KL penalty during our DPO training. Through this metric, we found that reducing $\beta$ from 0.3 to 0.2 increased the model's predictive uncertainty by 35%(0.3 in token-level-KL) relative to the SFT baseline. This quantitative leap in entropy correlates directly with the observed linguistic degradation (e.g., code-switching artifacts), marking the threshold of model stability.
