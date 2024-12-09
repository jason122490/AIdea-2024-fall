export CUDA_VISIBLE_DEVICES=0

seq_len=8192
patch_len=64
stride=32
model=LLM
model_name=EleutherAI/pythia-14m
embed_mode=Inverted
dataset_dir=Tbrain
dataset_name=Tbrain
model_des=pythia-14m

if [ ! -d "./logs/$dataset_name" ]; then
    mkdir ./logs/$dataset_name
fi

for pred_len in 1 
do
    python -u run.py \
      --is_training 0 \
      --root_path ./dataset/$dataset_dir/ \
      --data_path ../upload.csv \
      --exp_des $dataset_name'_'$seq_len'_'$pred_len \
      --model_des $model_des \
      --model $model \
      --model_name $model_name \
      --patch_len $patch_len \
      --stride $stride \
      --pretrained \
      --data Tbrain \
      --features MS \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --embed_mode $embed_mode
done