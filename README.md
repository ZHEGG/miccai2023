# The second place of MICCAI 2023 competition

Please refer to miccai2023 for data preparation and processing: https://github.com/LMMMEng/LLD-MMRI2023

# Overview
Our method integrates multiple models trained by cross-validation, and each model use class-balanced loss, oversampling strategy and data augment to alleviate the problems of data imbalance and overfitting in this competition

# Train
We provide our own txt files about five-fold cross-validation in ./data

The command for training is

    bash train.sh

Modifications of the relevant parameters can be implemented in the train.sh file

First

we select 6 models for integration, where fold3 was selected twice because it is difficult, which only requires simple modification of the parameter --model

fold1 for uniformer-base and its confusion matrix as follows

![Image text](https://github.com/ZHEGG/miccai2023/blob/main/image/fold1_confusion_matrix_base.png)

fold2 for uniformer-small and its confusion matrix as follows

![Image text](https://github.com/ZHEGG/miccai2023/blob/main/image/fold2_confusion_matrix_small.png)

fold3 for uniformer-small and uniformer-base and its confusion matrix as follows

![Image text](https://github.com/ZHEGG/miccai2023/blob/main/image/fold3_confusion_matrix_small.png)

![Image text](https://github.com/ZHEGG/miccai2023/blob/main/image/fold3_confusion_matrix_base.png)

fold4 for uniformer-base and its confusion matrix as follows

![Image text](https://github.com/ZHEGG/miccai2023/blob/main/image/fold4_confusion_matrix_base.png)

fold5 for uniformer-small and its confusion matrix as follows

![Image text](https://github.com/ZHEGG/miccai2023/blob/main/image/fold5_confusion_matrix_small.png)

second 

we use all trainval data to chose a model between 100 and 200 epochs based on previous training experience, There will be some randomness, and we will give the model weight of the competition submission

# Evaluation
The first output json is ensemble model

command

    predict_submit.py --data_dir {your data path} --val_anno_file root_dir/labels_test_inaccessible.txt --model uniformer_base_original uniformer_small_original uniformer_small_original uniformer_base_original uniformer_base_original uniformer_small_original --checkpoint {fold1.pth.tar fold2.pth.tar fold3.pth.tar fold3.pth.tar fold4.pth.tar fold5.pth.tar} --batch-size 1 --results-dir {your result dir} --team_name {your name} --img_size 16 128 128 --crop_size 14 112 112

The second output json is the model trained by all trainval data

just change --model and --checkpoint

finally we will get the final json by post-processing 2 model outputs using

    postprocess/json_refine.py

# Pretrained model
The pretrained model can be downloaded by:

https://drive.google.com/drive/folders/1kyspifcbd48-5FFnkJ9rgUZxIEUq8278?usp=drive_link