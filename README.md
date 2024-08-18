## Introduction

This repo contains the code and configurations (Raja's part) for our 3rd place solution in [LMSYS - Chatbot Arena Human Preference Prediction](https://www.kaggle.com/competitions/lmsys-chatbot-arena/overview) competition. 

- The summary of the solution is posted [here](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766).
- The inference notebook is posted [here](https://www.kaggle.com/code/conjuring92/lmsys-3rd-place-submission). 

Please refer to the following sections for details on training and dependencies.

## Section 1: Setup
### 1.1 Hardware
**vast.ai** was the primary source of compute. Specifically, models were trained on the following instance:

Ubuntu 22.04.3 LTS (128 GB boot disk)
AMD EPYC 75F3 32-Core Processor (128 vCPUs)
8 x NVIDIA RTX 6000 Ada

### 1.2 Software
I used PyTorch-2.2.1 image from vast.ai, which comes with:
* Python 3.10.13
* CUDA 12.4

### 1.3 Dependencies
Please clone the repository and install the required packages using the following commands:

```
git clone https://github.com/rbiswasfc/lmsys-arena.git
cd lmsys-arena
pip install -r requirements.txt
```

### 1.4 Datasets

Please make sure Kaggle API is installed. Then run the following script to download the required datasets:

```
chmod +x ./setup.sh
./setup.sh
```

Please note that the above script will create `datasets` and `models` folder in the directory located one level above the current directory. The external datasets will be downloaded in the `datasets` folder. Instruction-tuned LLMs, which can be used to generate adversarial essays, will be downloaded in the `models` folder. Total size of downloaded data and model files is ~8GB. 


## Section 2: Training
Training scripts and configurations are located in the `code` and `conf` folders respectively. We leveraged HF `accelerate` library to execute training runs with DDP on multiple GPUs (4x A100). Specifically, we used the following configurations for training:

```yaml
compute_environment: LOCAL_MACHINE                                            
debug: false                                                                           
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### 2.1 LLaMa
For (Q)LoRA fine-tuning of the LLaMa models:

```bash
accelerate launch ./lmsys/code/train_r_llama.py \
--config-name conf_r_llama_m205 \
seed=91 \
train_params.eval_frequency=500 \
debug=false \
save_model=true \
use_wandb=false
```
Please note that training takes ~12 hours. 

### 2.2 Gemma
For (Q)LoRA fine-tuning of the Gemma models:

```bash
accelerate launch ./lmsys/code/train_r_gemma.py \
--config-name conf_r_gemma_m110 \
seed=91 \
train_params.eval_frequency=500 \
debug=false \
save_model=true \
use_wandb=false
```

```bash
accelerate launch ./lmsys/code/train_r_gemma.py \
--config-name conf_r_gemma_m205 \
seed=91 \
train_params.eval_frequency=500 \
debug=false \
save_model=true \
use_wandb=false
```
Please note that training takes ~20 hours. 

