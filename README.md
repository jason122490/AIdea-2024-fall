# AIdea-2024-fall
AI CUP 2024 秋季賽 根據區域微氣候資料預測發電量競賽
## Requirement
+ OS: ubuntu 22.04
+ Nvidia GPU with CUDA version 12.2
+ Anaconda
## Environment Installation
Create an new conda virtual environment
```
conda create -n AIdea python=3.10 -y
conda activate AIdea
```
Install required packages:
```
pip install torch numpy pandas scikit-learn matplotlib reformer-pytorch
pip install timm transformers tqdm
```
Download code:
```
git clone https://github.com/jason122490/AIdea-2024-fall
cd AIdea-2024-fall
```
## Train and Infer step:
### 1. Training
Start training by running:
```
./scripts/Tbrain/<training setting>/LLM.sh
```
model weight and training log will be save in
```
./checkpoints/<training setting>/checkpoint.pth
./logs/Tbrain/<training setting>.log
```
### 2. Predict
Start predicting by running:
```
./scripts/Tbrain/<training setting>/LLM_pred.sh
```
private and public's prediction will be save in:
```
./results/<training setting>/results.csv
```
