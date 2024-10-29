# Model List
You can only use the following model checkpoints to initialize your LLM agent:

## Tested Models
We have tested the following models: (see **"Baseline Performance"** below for more details)
* Qwen/Qwen2.5-7B-Instruct
* meta-llama/Llama-3.1-8B-Instruct
* prince-canuma/Ministral-8B-Instruct-2410-HF

## Not Tested Models
* Qwen/Qwen2.5-1.5B-Instruct
* Qwen/Qwen2.5-0.5B-Instruct
* meta-llama/Llama-3.2-3B-Instruct
* meta-llama/Llama-3.2-1B-Instruct
* mistralai/Ministral-8B-Instruct-2410
* mistralai/Mistral-7B-Instruct-v0.3
* google/gemma-2-2b-it
* google/gemma-2-9b-it

## Baseline Performance
Here are the baseline performance (and total inference time in minutes on a single NVIDIA 3090 GPU) of different models for the public datasets. (Currently, we only tested the following 3 models)
* Qwen/Qwen2.5-7B-Instruct

| Task Type     | Classification (max_token = 16) | SQL Generation (max_token = 512) |
|---------------|---------------------------------|----------------------------------|
| Dataset       | DDXPlus                   | BIRD                   |
| Zero-shot     | 43.48% (~12.5 m)                 | 26.08% (~63.9 m)                 |
| Self-StreamICL| 51.53% (~49.0 m)                 | 30.64% (~62.0 m)                 |

* meta-llama/Llama-3.1-8B-Instruct

| Task Type      | Classification (max_token = 16) | SQL Generation (max_token = 512) |
|----------------|---------------------------------|----------------------------------|
| Dataset        | DDXPlus                   | BIRD                   |
| Zero-shot      | 44.95% (~26.5 m)                 | 26.08% (~78.1 m)                 |
| Self-StreamICL | 49.38% (~57.5 m)                 | 30.77% (~82.5 m)                 |

* prince-canuma/Ministral-8B-Instruct-2410-HF

| Task Type      | Classification (max_token = 16) | SQL Generation (max_token = 512) |
|----------------|---------------------------------|----------------------------------|
| Dataset        | DDXPlus                   | BIRD                   |
| Zero-shot      | 29.48% (~30.6 m)                 | 22.36% (~86.3 m)                 |
| Self-StreamICL | 39.29% (~58.7 m)                 | 29.27% (~90.5 m)                 |
