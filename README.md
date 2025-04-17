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

![image](https://github.com/user-attachments/assets/14ee9f36-fee1-4f7e-83b9-c0512f5eafae)


Here is the recommended citation for our paper in different common formats:

MLA:  Alp, Sait, et al. "Joint transformer architecture in brain 3D MRI classification: its application in Alzheimer’s disease classification." Scientific Reports 14.1 (2024): 8996.

APA:  Alp, S., Akan, T., Bhuiyan, M. S., Disbrow, E. A., Conrad, S. A., Vanchiere, J. A., ... & Bhuiyan, M. A. (2024). Joint transformer architecture in brain 3D MRI classification: its application in Alzheimer’s disease classification. Scientific Reports, 14(1), 8996.

Chicago:  Alp, Sait, Taymaz Akan, Md Shenuarin Bhuiyan, Elizabeth A. Disbrow, Steven A. Conrad, John A. Vanchiere, Christopher G. Kevil, and Mohammad AN Bhuiyan. "Joint transformer architecture in brain 3D MRI classification: its application in Alzheimer’s disease classification." Scientific Reports 14, no. 1 (2024): 8996.

Harvard:  Alp, S., Akan, T., Bhuiyan, M.S., Disbrow, E.A., Conrad, S.A., Vanchiere, J.A., Kevil, C.G. and Bhuiyan, M.A., 2024. Joint transformer architecture in brain 3D MRI classification: its application in Alzheimer’s disease classification. Scientific Reports, 14(1), p.8996.

Vancouver:  Alp S, Akan T, Bhuiyan MS, Disbrow EA, Conrad SA, Vanchiere JA, Kevil CG, Bhuiyan MA. Joint transformer architecture in brain 3D MRI classification: its application in Alzheimer’s disease classification. Scientific Reports. 2024 Apr 18;14(1):8996.
