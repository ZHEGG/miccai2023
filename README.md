# The second place of MICCAI 2023 competition

Please refer to miccai2023 for data preparation and processing: https://github.com/LMMMEng/LLD-MMRI2023

# Overview
Our method integrates multiple models trained by cross-validation, and each model use class-balanced loss, oversampling strategy and data augment to alleviate the problems of data imbalance and overfitting in this competition

![Image text](https://github.com/ZHEGG/miccai2023/blob/main/image/pipeline.png)

# Train
We provide our own txt files about five-fold cross-validation in ./data

The command for training is

    bash train.sh

Modifications of the relevant parameters can be implemented in the train.sh file

First

We select 6 models for integration, where fold3 was selected twice because it is difficult, which only requires simple modification of the parameter --model

Fold1 for uniformer-base and its confusion matrix as follows

![Image text](https://github.com/ZHEGG/miccai2023/blob/main/image/fold1_confusion_matrix_base.png)

Fold2 for uniformer-small and its confusion matrix as follows

![Image text](https://github.com/ZHEGG/miccai2023/blob/main/image/fold2_confusion_matrix_small.png)

Fold3 for uniformer-small and uniformer-base and its confusion matrix as follows

![Image text](https://github.com/ZHEGG/miccai2023/blob/main/image/fold3_confusion_matrix_small.png)

![Image text](https://github.com/ZHEGG/miccai2023/blob/main/image/fold3_confusion_matrix_base.png)

Fold4 for uniformer-base and its confusion matrix as follows

![Image text](https://github.com/ZHEGG/miccai2023/blob/main/image/fold4_confusion_matrix_base.png)

Fold5 for uniformer-small and its confusion matrix as follows

![Image text](https://github.com/ZHEGG/miccai2023/blob/main/image/fold5_confusion_matrix_small.png)

Second 

We use all trainval data to train a model, we save model between 100 and 250 epochs every 10 epochs based on previous training experience, and save a best model according to loss, Then we choose one of them, we also give the model weight of the competition submission

The command for training is

    bash train_alldata.sh
# Evaluation
The first output json is trained by ensemble model

Command

    python predict_submit.py --data_dir {your data path} --val_anno_file root_dir/labels_test_inaccessible.txt --model uniformer_base_original uniformer_small_original uniformer_small_original uniformer_base_original uniformer_base_original uniformer_small_original --checkpoint {fold1.pth.tar fold2.pth.tar small_fold3.pth.tar base_fold3.pth.tar fold4.pth.tar fold5.pth.tar} --batch-size 1 --results-dir {your result dir} --team_name {your name} --img_size 16 128 128 --crop_size 14 112 112

The second output json is the model trained by all trainval data

Just change the parameter --model {uniformer_small_original} and give the corresponding --checkpoint {trainval.pt} and add --modified

Command

    python predict_submit.py --data_dir {your data path} --val_anno_file root_dir/labels_test_inaccessible.txt --model uniformer_small_original --checkpoint {trainval.pt} --batch-size 1 --results-dir {your result dir} --team_name {your name} --img_size 16 128 128 --crop_size 14 112 112 --modified

Finally we will get the final json by post-processing 2 model outputs using

    python postprocess/json_refine.py

Noted that json trained by ensemble should be put in first and json trained by all trainval data should be put in second

# Pretrained model
The pretrained model can be downloaded by:

https://drive.google.com/drive/folders/1kyspifcbd48-5FFnkJ9rgUZxIEUq8278?usp=drive_link

The pretrained model for uniformer all can be found in:

https://github.com/Sense-X/UniFormer/tree/main/video_classification