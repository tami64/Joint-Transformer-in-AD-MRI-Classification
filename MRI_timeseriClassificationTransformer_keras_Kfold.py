'''
This code was written by Dr. Sait Alp and Dr. Taymaz Akan for our paper:
Alp, S., Akan, T., Bhuiyan, M.S., Disbrow, E.A., Conrad, S.A., Vanchiere, J.A., Kevil, C.G., & Bhuiyan, M.A. (2024). 
"Joint transformer architecture in brain 3D MRI classification: its application in Alzheimer’s disease classification". Scientific Reports, 14(1), 8996.
https://www.nature.com/articles/s41598-024-59578-3
Please cite this paper if you use this code.
'''

import numpy as np
from numpy import mean
from numpy import std
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import os
from pickle import load

BATCH_SIZE = 32
EPOCHS = 100
K_Fold_Size = 10
MAX_SEQ_LENGTH = 50
desired_lbl = ['AD','CN','MCI'] #
planes = ['Axial','Coronal','Sagittal']
plane = planes[1]

folderName = 'features_ViT_CAT12_'+ plane + '_' +str(MAX_SEQ_LENGTH)+'Slices'
features_path = os.path.join('ADNI1_Complete_1Yr_1.5T_CAT12_Planes',folderName)

features_path_k_Fold = os.path.join(features_path, 'K_Fold')

# load the label_encoder model
label_encoder = load(open(features_path_k_Fold +'/lblEncoder.pkl', 'rb'))

className = np.load(features_path_k_Fold + '/className.npy')
n_classes = len(desired_lbl)
if(n_classes == 2):
    className = np.asarray(desired_lbl)

#----------------------------------------------------------------------------------------------------------------------#
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(x) #
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

#----------------------------------------------------------------------------------------------------------------------#
# Utility for our sequence model.
def train_test_model(x_train, y_train, x_test, y_test,x_valid, y_valid, filepath,fold_no):
    input_shape = x_train.shape[1:]

    model = build_model(
        input_shape,
        head_size=32,
        num_heads=10,
        ff_dim=512,
        num_transformer_blocks=8,
        mlp_units=[100, 30],  #
        mlp_dropout=0.15,
        dropout=0.15,

    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )

    print(model.summary())
    print(len(model.layers))

    if (n_classes == 2):
        model_name = "best_model_ViT_Transformer_binary"+ plane +"_fold" + str(fold_no) + ".h5"
    else:
        model_name = "best_model_ViT_Transformer"+ plane +"_fold" + str(fold_no) + ".h5"

    model_path =  os.path.join(filepath, model_name)

    checkpoint = keras.callbacks.ModelCheckpoint(
        model_path, save_best_only=True, verbose=1
    )
    erlyStop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    callbacks = [checkpoint]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True
    )


    # evaluate over best model
    savedModel = keras.models.load_model(model_path)
    score = savedModel.evaluate([x_test], y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    y_preds_prob = savedModel.predict([x_test])
    y_preds = np.argmax(y_preds_prob, axis=1)
    report = classification_report(y_test, y_preds, target_names=className)
    Con_fold = tf.math.confusion_matrix(y_test, y_preds, num_classes=n_classes)
    print(report)  # , target_names=className
    return score[1],Con_fold,y_test,y_preds,report

#----------------------------------------------------------------------------------------------------------------------#
def Convert2BinaryProblem(data,lbls,desired_lbl,className):
    index = []
    for item in desired_lbl:
        item_int = np.where(className == item)
        item_int2 = label_encoder.transform([item])
        index_t = np.where(lbls == item_int2[0])[0]
        index.extend(index_t)

    lbls_b = lbls[index]
    data_b = data[index]
    return data_b,lbls_b
#----------------------------------------------------------------------------------------------------------------------#

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
#----------------------------------------------------------------------------------------------------------------------#
# Utility for running experiments.
def run_experiment(K_Fold):
    filepath = os.path.join(features_path_k_Fold, 'savedModels')
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
        print('Create new  ' + filepath + '  folder')

    if (n_classes == 2):
        txt_file_name = 'ViT_Transformer_CAT12_report_binary_' + plane + '.txt'
    else:
        txt_file_name = 'ViT_Transformer_CAT12_report_' + plane + '.txt'


    f = open(txt_file_name, 'w')
    scores = list()
    all_fold_predicted_lbl  = list()
    all_fold_actual_lbl = list()

    allConfMat = np.zeros((n_classes, n_classes))
    for fold_no in range(0,K_Fold):  # stratSplit.split(x, y):
        fold_no += 1

        fileName = "traindata_Fold" + str(fold_no) + ".npy"
        trainX = np.load(os.path.join(features_path_k_Fold,fileName))
        fileName = "trainlbl_Fold" + str(fold_no) + ".npy"
        trainy = np.load(os.path.join(features_path_k_Fold,fileName))

        fileName = "testdata_Fold" + str(fold_no) + ".npy"
        testX = np.load(os.path.join(features_path_k_Fold,fileName))
        fileName = "testlbl_Fold" + str(fold_no) + ".npy"
        testy = np.load(os.path.join(features_path_k_Fold,fileName))

        fileName = "validdata_Fold" + str(fold_no) + ".npy"
        validX = np.load(os.path.join(features_path_k_Fold,fileName))
        fileName = "validlbl_Fold" + str(fold_no) + ".npy"
        validy = np.load(os.path.join(features_path_k_Fold,fileName))

        if(n_classes == 2):
            # -----------------------Convert2BinaryProblem----------------------#
            trainX, trainy = Convert2BinaryProblem(trainX, trainy, desired_lbl, className)
            testX, testy = Convert2BinaryProblem(testX, testy, desired_lbl, className)
            validX, validy = Convert2BinaryProblem(validX, validy, desired_lbl, className)


        #----------------------------Scaler Data----------------------------#
        (NUM_SCANS, MAX_SEQ_LENGTH, NUM_FEATURES) = trainX.shape
        seq = np.reshape(trainX, (NUM_SCANS, NUM_FEATURES * MAX_SEQ_LENGTH))
        cs = StandardScaler()
        data_normalized = cs.fit_transform(seq)
        trainX = np.reshape(data_normalized, (NUM_SCANS, MAX_SEQ_LENGTH, NUM_FEATURES))

        (NUM_SCANS, MAX_SEQ_LENGTH, NUM_FEATURES) = testX.shape
        seq = np.reshape(testX, (NUM_SCANS, NUM_FEATURES * MAX_SEQ_LENGTH))
        data_normalized = cs.transform(seq)
        testX = np.reshape(data_normalized, (NUM_SCANS, MAX_SEQ_LENGTH, NUM_FEATURES))

        (NUM_SCANS, MAX_SEQ_LENGTH, NUM_FEATURES) = validX.shape
        seq = np.reshape(validX, (NUM_SCANS, NUM_FEATURES * MAX_SEQ_LENGTH))
        data_normalized = cs.transform(seq)
        validX = np.reshape(data_normalized, (NUM_SCANS, MAX_SEQ_LENGTH, NUM_FEATURES))

        score_fold,Con_fold,y_test,y_preds,report = train_test_model(trainX, trainy, testX, testy, validX, validy, filepath, fold_no)
        score_fold = score_fold * 100.0
        print('>#%d: %.2f' % (fold_no, score_fold))
        scores.append(score_fold)
        all_fold_actual_lbl.extend(y_test)
        all_fold_predicted_lbl.extend(y_preds)
        allConfMat += np.array(Con_fold)
        f.write('Fold: '+ str(fold_no) +'\n')
        f.write(report + '\n')
        f.write('\n')

    f.close()
    # summarize results
    summarize_results(scores)

    #-----------------------------------------Plot All K-Fold Confusion Matrix-----------------------------------------#
    fig, ax = plot_confusion_matrix(conf_mat=allConfMat,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True,
                                    class_names=className,
                                    figsize=(12, 6))
    if (n_classes == 2):
        fname = 'ConfusionMat_AllFold_ViT_Transformer_Plane_' + plane + '_binary'
    else:
        fname = 'ConfusionMat_AllFold_ViT_Transformer_Plane_' + plane

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14}

    matplotlib.rc('font', **font)

    fig.savefig(fname)
    sumElement = np.sum(allConfMat)
    accuracy = np.sum(allConfMat.diagonal()) / sumElement
    print('Accuracy: ', accuracy)
    #plt.show()

    report = classification_report(all_fold_actual_lbl, all_fold_predicted_lbl, target_names=className)
    print('Performans metrics for all folds:')
    print(report)
    f = open(txt_file_name, 'a')
    f.write('\n ------------------------------------------\n')
    f.write('\nPerformans metrics for all folds:\n\n')
    f.write(report)
    f.write('\n ------------------------------------------\n')
    f.write(repr(scores))
    m, s = mean(scores), std(scores)
    f.write('\nAccuracy: %.3f%% (+/-%.3f)' % (m, s))
    f.close()

#----------------------------------------------------------------------------------------------------------------------#
run_experiment(K_Fold_Size)
