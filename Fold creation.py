import os
import nibabel as nib
from glob2 import glob
from tqdm import tqdm
import cv2
import imageio

'''for creating equals folds'''

org_img = sorted(glob('D:\Covid noisy dataset\clean data\Fold for train\jpg image\*'))
mask_infected = sorted(glob('D:\Covid noisy dataset\clean data\Fold for train\jpg mask\*'))

output_path1_jpg = 'D:\Covid noisy dataset\clean data\Fold for train\Fold1\jpg image'
output_path1_mask = 'D:\Covid noisy dataset\clean data\Fold for train\Fold1\jpg mask'

output_path2_jpg = 'D:\Covid noisy dataset\clean data\Fold for train\Fold2\jpg image'
output_path2_mask = 'D:\Covid noisy dataset\clean data\Fold for train\Fold2\jpg mask'

output_path3_jpg = 'D:\Covid noisy dataset\clean data\Fold for train\Fold3\jpg image'
output_path3_mask = 'D:\Covid noisy dataset\clean data\Fold for train\Fold3\jpg mask'

output_path4_jpg = 'D:\Covid noisy dataset\clean data\Fold for train\Fold4\jpg image'
output_path4_mask = 'D:\Covid noisy dataset\clean data\Fold for train\Fold4\jpg mask'

number_of_folds = 4

j = 0
for i in tqdm(range(len(org_img))):
    patient = org_img[i]
    patient_mask = mask_infected[i]
    patient_name = os.path.basename(os.path.normpath(patient))
    image = cv2.imread(patient)
    mask = cv2.imread(patient_mask)
    if j == 0:
        imageio.imwrite(os.path.join(output_path1_jpg, patient_name.replace('.nii', '')),
                        image)

        imageio.imwrite(os.path.join(output_path1_mask, patient_name.replace('.nii', '')),
                        mask)
    if j == 1:
        imageio.imwrite(os.path.join(output_path2_jpg, patient_name.replace('.nii', '')),
                        image)

        imageio.imwrite(os.path.join(output_path2_mask, patient_name.replace('.nii', '')),
                        mask)
    if j == 2:
        imageio.imwrite(os.path.join(output_path3_jpg, patient_name.replace('.nii', '')),
                        image)

        imageio.imwrite(os.path.join(output_path3_mask, patient_name.replace('.nii', '')),
                        mask)
    if j == 3:
        imageio.imwrite(os.path.join(output_path4_jpg, patient_name.replace('.nii', '')),
                        image)

        imageio.imwrite(os.path.join(output_path4_mask, patient_name.replace('.nii', '')),
                        mask)
    if (i+1) % (len(org_img)/number_of_folds) == 0:
        j = j+1
