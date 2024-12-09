export CUDA_VISIBLE_DEVICES=0

seq_len=4096
patch_len=32
stride=16
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
      --is_training 1 \
      --root_path ./dataset/$dataset_dir/ \
      --exp_des $dataset_name'_'$seq_len'_'$pred_len \
      --model_des $model_des \
      --model $model \
      --model_name $model_name \
      --patch_len $patch_len \
      --stride $stride \
      --pretrained \
      --cached_dataset \
      --data Tbrain \
      --features MS \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --embed_mode $embed_mode \
      --itr 1 \
      --train_epochs 50 --patience 50 --batch_size 128 --learning_rate 0.0001 --weight_decay 0.01 \
       > logs/$dataset_name/$dataset_name'_'$seq_len'_'$pred_len'_'$model_des'_'$patch_len'_'$stride'_pretrained.log'
done