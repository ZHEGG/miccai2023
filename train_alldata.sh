#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"

PORT=${PORT:-49500}

python3 -m torch.distributed.launch --master_port=$PORT --nproc_per_node=2 train_alldata.py \
  --data_dir /home/mdisk3/bianzhewu/dataset/医疗数据集/miccai2023/classification_dataset/images/ \
  --train_anno_file /home/mdisk3/bianzhewu/dataset/医疗数据集/miccai2023/classification_dataset/labels/labels_trainval.txt \
  --batch-size 4 \
  --model uniformer_small_original \
  --lr 1e-4 \
  --warmup-epochs 5 \
  --epochs 300 \
  --output output/ \
  --train_transform_list random_crop z_flip x_flip y_flip rotation edge emboss filter \
  --crop_size 14 112 112 \
  --pretrained \
  --sampling sqrt \
  --mixup \
  --cb_loss \
  --smoothing 0.1 \
  --img_size 16 128 128 \
  --drop-path 0.1 \
  --eval-metric loss
  # --mode tricubic 
  # --train_anno_file /home/mdisk3/bianzhewu/dataset/医疗数据集/miccai2023/classification_dataset/labels/labels.txt \
  
  

