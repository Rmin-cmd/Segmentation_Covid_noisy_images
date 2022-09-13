# Segmentation Covid noisy images

In this repository, Standardization techniques(Histogram Standardization and Z Normalization) was examined to check whether these techniques could improve segmetation metrics and help 
to segment noisy lung images three different classes:

1.background class(class 0)

2.lung region(class 1)

3.infected area(class 2)

# Dataset 

we have used Zenodo dataset contains 20 niffti images and half of them are in range [0,255] and remaining are in range [-1000, 1000].

Link: https://zenodo.org/record/3757476

First, we added noise to these dataset and for examining effect of standardization technique, 4 datasets were made:

1. Low Noise Dataset (additive White Gaussian Noise with stadard deviation of 8)

2. Standardize Low Noise Dataset

3. High Noise Dataset (additive White Gaussian Noise with Standard deviation of 40)

4. Standardize High Noise Dataset 

# Segmentation Model and Framework

In this experiment, we used Pytorch as our base framework and Segmentation_models_pytorch library for choosing suitable model. Unet++ was used with RegNetx_004 as backbone.

This model was showed to have be among best models for segmetation infected lung regions with F-Score:0.8487 and IOU:0.8138 by https://doi.org/10.48550/arXiv.2205.09722.

we trained for 50 epochs and ImageNet weights was initialized for Transfer Learning

# HyperParameter

all the images were resized to 256 x 256.

learning rate : 5e-4

Batch Size: 8

Five Fold Cross Validation

Adam Optimizer 

GPU : NVIDIA Geforce RTX 3060

# Results
Results and furthur details reported in this paper
