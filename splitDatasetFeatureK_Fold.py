'''
This code was written by Dr. Sait Alp and Dr. Taymaz Akan for our paper:
Alp, S., Akan, T., Bhuiyan, M.S., Disbrow, E.A., Conrad, S.A., Vanchiere, J.A., Kevil, C.G., & Bhuiyan, M.A. (2024). 
"Joint transformer architecture in brain 3D MRI classification: its application in Alzheimer’s disease classification". Scientific Reports, 14(1), 8996.
https://www.nature.com/articles/s41598-024-59578-3
Please cite this paper if you use this code.
'''
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import itertools
import os
from pickle import dump


MAX_SEQ_LENGTH = 50
planes = ['Axial','Coronal','Sagittal']
plane = planes[2]

folderName = 'features_ViT_CAT12_'+ plane + '_' +str(MAX_SEQ_LENGTH)+'Slices'
features_path = os.path.join('ADNI1_Complete_1Yr_1.5T_CAT12_Planes',folderName)


scan_data = np.load( features_path + '/scan_data.npy')
scan_mask = np.load( features_path + '/scan_mask.npy')
scan_labels = np.load( features_path + '/scan_labels.npy')

features_path_k_Fold = os.path.join(features_path, 'K_Fold')
if not os.path.isdir(features_path_k_Fold):
    os.makedirs(features_path_k_Fold)
    print('Create new  ' + features_path_k_Fold + '  folder')


label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(scan_labels)
print(label_encoder.classes_)
className = np.array(label_encoder.classes_)
np.save(features_path_k_Fold + '/className' , className)

# save the label_encoder
dump(label_encoder, open(features_path_k_Fold +'/lblEncoder.pkl', 'wb'))

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
for i, (train_index, test_index) in enumerate(sss.split(scan_data, scan_labels)):
    print(f"Fold {i}:")
    traindataTmp = scan_data[train_index]
    trainlblTmp = labels[train_index]

    testdata = scan_data[test_index]
    testlbl = labels[test_index]

    foldName = i + 1
    file_path =  os.path.join(features_path_k_Fold, 'testdata_Fold'+ str(foldName))
    np.save(file_path, testdata)
    file_path = os.path.join(features_path_k_Fold, 'testlbl_Fold' + str(foldName))
    np.save(file_path, testlbl)

    # split validation data from train
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for j, (train_index, val_index) in enumerate(sss.split(traindataTmp, trainlblTmp)):
        traindata = traindataTmp[train_index]
        trainlbl = trainlblTmp[train_index]
        validdata = traindataTmp[val_index]
        validlbl = trainlblTmp[val_index]

        file_path = os.path.join(features_path_k_Fold, 'traindata_Fold' + str(foldName))
        np.save(file_path, traindata)
        file_path = os.path.join(features_path_k_Fold, 'trainlbl_Fold' + str(foldName))
        np.save(file_path, trainlbl)
        file_path = os.path.join(features_path_k_Fold, 'validdata_Fold' + str(foldName))
        np.save(file_path, validdata)
        file_path = os.path.join(features_path_k_Fold, 'validlbl_Fold' + str(foldName))
        np.save(file_path, validlbl)