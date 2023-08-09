# The second place of MICCAI 2023 competition

Please refer to miccai2023 for data preparation and processing: https://github.com/LMMMEng/LLD-MMRI2023

# Overview
Our method integrates multiple models trained by cross-validation, and each model use class-balanced loss, oversampling strategy and data augment to alleviate the problems of data imbalance and overfitting in this competition

# Train
We provide our own txt files about five-fold cross-validation in ./data

The command for training is 
bash train.sh

Modifications of the relevant parameters can be implemented in the train.sh file

we select 6 models for integration, where fold3 was selected twice because it is difficult, which only requires simple modification of the parameter --model

fold1 for uniformer-base and its confusion matrix as follows
