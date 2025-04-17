'''
This code was written by Dr. Sait Alp and Dr. Taymaz Akan for our paper:
Alp, S., Akan, T., Bhuiyan, M.S., Disbrow, E.A., Conrad, S.A., Vanchiere, J.A., Kevil, C.G., & Bhuiyan, M.A. (2024). 
"Joint transformer architecture in brain 3D MRI classification: its application in Alzheimer’s disease classification". Scientific Reports, 14(1), 8996.
https://www.nature.com/articles/s41598-024-59578-3
Please cite this paper if you use this code.
'''
import os
import pandas as pd
import numpy as np
import cv2

import shutil
import glob
from tempfile import TemporaryFile
from numpy import save
from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths
from transformers import TFViTForImageClassification
from transformers import ViTFeatureExtractor, TFViTModel

#----------------------------------load .csv file----------------------------#
dataset_dir = 'ADNI1_Complete_1Yr_1.5T_CAT12_Planes/1Y1.5T_CAT12'  #preprocessed with CAT12
dataset_labelfile = 'ADNI1_Complete_1Yr_1.5T_CAT12_Planes/ADNI1_Complete_1Yr_1.5T.csv'
dataset_labelfile_idfield = 'Image Data ID'    #field name in .csv file
dataset_labelfile_labelfield = 'Group'         #field name in .csv file

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = TFViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

IMG_SIZE = 224 #slice size 256*256
BATCH_SIZE = 25
EPOCHS = 10

MAX_SEQ_LENGTH = 50 #maximum slice for each MRI scan
NUM_FEATURES = 768

planes = ['Axial','Coronal','Sagittal']
plane = planes[2]

dataset_dir = dataset_dir + plane
#-----------------------------------------------------------------------------#
def df2dic(df):
    L1 = df.get('Image Data ID')
    L2 = df.get('Group')
    thisdict = dict(zip(L1, L2))
    return thisdict
#-----------------------------------------------------------------------------#

all_data_df = pd.read_csv(dataset_labelfile)
print(f"Total scans: {len(all_data_df)}")
all_label_dc = df2dic(all_data_df)

#-----------------------------------------------------------------------------#
def build_feature_extractor(image_):
    image = image_[0, :, :, :]
    inputs = feature_extractor(images=image, return_tensors="tf")
    outputs = model(**inputs)
    features = outputs.pooler_output
    return features.numpy()

#-----------------------------------------------------------------------------#

def load_scan_images(scan_path, resize=(IMG_SIZE, IMG_SIZE)):
    slices = []
    path = scan_path + '/*.png'
    listOfSlicesPath = glob.glob(path, recursive=True)
    slices_nums = len(listOfSlicesPath)
    if(slices_nums <= MAX_SEQ_LENGTH ):
        startIndex = 0
        stopIndex = slices_nums
    else:
        diff = abs(slices_nums - MAX_SEQ_LENGTH)
        startIndex = diff//2
        stopIndex = slices_nums - (diff//2)

    for i in range(startIndex,stopIndex,1):   #position to stop (not included)
        slicePath = os.path.join(scan_path ,str(i)+'.png')
        img = cv2.imread(slicePath, cv2.IMREAD_COLOR)
        slice = cv2.resize(img, resize)
        slice = slice[:, :, [2, 1, 0]]
        slices.append(slice)

    return np.array(slices)
#-----------------------------------------------------------------------------#

def prepare_all_MRI_scans(all_label_dc,dataset_dir):
    # absolute path to search all  files inside a specific folder
    listOfScanId = os.listdir(dataset_dir)
    num_samples = len(listOfScanId)
    labels = []

    # Initialize placeholders to store the masks and features of the current scan.
    frame_masks  = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )
    idx = 0
    # For each scan.
    for scanId in listOfScanId:
        scan_path = os.path.join(dataset_dir, scanId)
        slices = load_scan_images(scan_path, resize=(IMG_SIZE, IMG_SIZE))
        slices = slices[None, ...]
        labels.append(all_label_dc[scanId])

        # Initialize placeholders to store the masks and features of the current scan.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the slices of the current scan.
        for i, batch in enumerate(slices):
            scan_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, scan_length)
            for j in range(length):
                temp_frame_features[i, j, :] = build_feature_extractor(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked


        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()
        idx += 1
        print("Extracte features for Scan: ",idx)
    return (frame_features, frame_masks), np.array(labels)
#-----------------------------------------------------------------------------#

all_scan_data, scan_labels = prepare_all_MRI_scans(all_label_dc,dataset_dir)

folderName = 'features_ViT_CAT12_'+ plane + '_' +str(MAX_SEQ_LENGTH)+'Slices'
dirName = os.path.join('ADNI1_Complete_1Yr_1.5T_CAT12_Planes',folderName)


if not os.path.exists(dirName):
           os.makedirs(dirName)

file_path = dirName+'/scan_data'
np.save(file_path,all_scan_data[0])
file_path = dirName+'/scan_mask'
np.save(file_path,all_scan_data[1])
file_path = dirName+'/scan_labels'
np.save(file_path,scan_labels)




