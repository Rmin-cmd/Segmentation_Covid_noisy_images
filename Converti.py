import os
import nibabel as nib
from glob2 import glob
from tqdm import tqdm
import cv2

''' this code is for converting niffti files to range 0 to 255'''
def saveniffti(img_np, name, affine):
    ni_img = nib.Nifti1Image(img_np, affine)
    nib.save(ni_img, os.path.join('D:\deep learning project\Data_kaggle\kaggle dataset\data kaggle rescaled',
                                  name+str('.gz')))


def saveniffti_mask(img_np, name, affine):
    ni_img = nib.Nifti1Image(img_np, affine)
    nib.save(ni_img, os.path.join('D:\deep learning project\Data_kaggle\kaggle dataset\mask kaggle rescaled',
                                  name+str('.gz')))


org_img = sorted(glob('D:\deep learning project\Data_kaggle\kaggle dataset\ct_scans\*'))
mask_infected = sorted(glob('D:\deep learning project\Data_kaggle\kaggle dataset\lung_and_infection_mask (1)\*'))

for i, patient in enumerate(tqdm(org_img)):
    img_org = nib.load(patient)
    aff = img_org.affine
    num_org = img_org.get_fdata()
    patient_name_img = os.path.basename(os.path.normpath(patient))

    img_infected = nib.load(mask_infected[i])
    aff_mask = img_infected.affine
    num_infected = img_infected.get_fdata()
    patient_name_mask = os.path.basename(os.path.normpath(mask_infected[i]))
    x = num_org.shape[2]

    for j in range(0, x):
        num_org[:, :, j] = cv2.normalize(num_org[:, :, j], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_32F)

    saveniffti(num_org, patient_name_img, aff)
    saveniffti_mask(num_infected, patient_name_mask, aff_mask)
