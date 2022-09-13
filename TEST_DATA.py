
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import imutils
import segmentation_models_pytorch as smp

def to_categorical(y,num_classes,dtype = 'float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path,test_flag,preprocessing=None,augmentation=None):

        self.images_path = images_path
        self.masks_path = masks_path
        self.test_flag = test_flag
        self.n_samples = len(images_path)
        self.counter = 0
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.num_classes = 3

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index])
        #image = nib.load(self.images_path[index]).get_fdata()
        image = cv2.resize(image,[256,256])
        image = image/255.0 ## (512, 512, 3)
        #image = np.reshape(image,[3,256,256])
        #image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        #image = image.astype(np.float32)

        #image = torch.tensor(image)
        self.counter = self.counter+1
        #print(self.counter)

        """ Reading mask """
        #mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = Image.open(self.masks_path[index]).convert('L')
        mask = mask.resize((256,256),Image.ANTIALIAS)
        height,width = mask.size[0],mask.size[1]
        mask = list(mask.getdata())
        mask = np.array(mask)
        mask = mask.reshape([height,width])
        #mask = imutils.resize(mask,256,256)
        #mask = nib.load(self.masks_path[index]).get_fdata()
        #print(mask.shape)
        #mask = cv2.resize(mask,[256,256])
        #print(np.max(mask*3))
        #mask = np.around((mask/255.0)*3)  ## (256, 256)
        #print(mask)
        mask = np.around(mask / 128.0)
        #print(np.unique(mask))
        #if self.test_flag == False:
            #mask = np.where(mask == 2,1,mask)
        #mask = np.where(mask == 2, 1, mask)
        #mask = np.where(mask == 3,2,mask)

        masks = np.zeros((height, width, self.num_classes))
        for i, unique_value in enumerate(np.unique(mask)):
            masks[:, :, int(unique_value)][mask == unique_value] = 1

        #mask = to_categorical(mask,3)

        #print(mask.shape)
        #mask = np.transpose(mask, (2, 0, 1))
        #mask = np.expand_dims(mask, axis=0) ## (1, 256, 256)

        #print(mask.shape)
        #mask = mask.astype(np.float32)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=masks)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        #print(image.shape,mask.shape)
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=masks)
            image, mask = sample['image'], sample['mask']
        #mask = torch.tensor(mask)
        #print(mask.shape)
        image = torch.from_numpy(image)
        masks = torch.from_numpy(mask)
        #mask = masks.to(dtype=torch.int64)

        return image, masks

    def __len__(self):
        return self.n_samples