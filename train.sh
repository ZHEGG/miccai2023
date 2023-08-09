#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5"

PORT=${PORT:-69500}

python3 -m torch.distributed.launch --master_port=$PORT --nproc_per_node=2 train.py \
  --data_dir /home/mdisk3/bianzhewu/dataset/医疗数据集/miccai2023/classification_dataset/allimages/ \
  --train_anno_file /home/mdisk3/bianzhewu/dataset/医疗数据集/miccai2023/classification_dataset/labels/labels_trainval.txt \
  --val_anno_file /home/mdisk3/bianzhewu/dataset/医疗数据集/miccai2023/classification_dataset/labels/labels_test_custom.txt \
  --batch-size 4 \
  --model uniformer_base_original \
  --lr 1e-4 \
  --warmup-epochs 5 \
  --epochs 300 \
  --output output/ \
  --train_transform_list random_crop z_flip x_flip y_flip rotation edge emboss filter \
  --crop_size 14 112 112 \
  --pretrained \
  --mixup \
  --cb_loss \
  --smoothing 0.1 \
  --img_size 16 128 128 \
  --drop-path 0.1 \
  --eval-metric f1 kappa
  # --mode tricubic 
  # --train_anno_file /home/mdisk3/bianzhewu/dataset/医疗数据集/miccai2023/classification_dataset/labels/labels.txt \
  # --val_anno_file /home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/output/labels.txt \
  # --sampling class \