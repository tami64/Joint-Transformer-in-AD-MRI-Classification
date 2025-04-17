# Joint-Transformer-in-AD-MRI-Classification
Joint Transformer Architecture in Brain 3D MRI Classification: Its Application in Alzheimer’s Disease Classification

To use this code for classifying MRI datasets:
Preprocess the MRI dataset using CAT12 tools.

Run extract_save_features_ViT.py
→ Extract and save ViT-based features from MRI slices in the desired planes ('Axial','Coronal','Sagittal').

Run splitDatasetFeatureK_Fold.py
→ Generate 10-fold cross-validation sets from the dataset.

Run MRI_timeseriClassificationTransformer_keras_Kfold.py
→ Perform classification using the transformer-based model on the 10-fold dataset.

