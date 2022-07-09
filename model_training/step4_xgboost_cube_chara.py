import settings
import tensorflow as tf
import helpers
import sys
from keras.layers import BatchNormalization
# import  keras.layers.BatchNormalization as BN
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import keras
from keras.optimizers import SGD,RMSprop,Adadelta,Adam,Adamax,Nadam
import os
import glob
import random
import pandas
import ntpath
import cv2
import numpy
from keras.utils import np_utils
from typing import List, Tuple
from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, Conv3D,UpSampling2D, merge, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import math
import shutil
from itertools import cycle
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# limit memory usage..
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import roc_curve, auc, confusion_matrix
import itertools
from scipy import interp
from keras import layers
from keras import models
import xgboost as xgb
from xgboost import plot_importance
from xgboost import plot_tree
import shap
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import train_test_split,StratifiedKFold,train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error,roc_auc_score
import seaborn as sns
from scipy.stats import pearsonr,spearmanr

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=config))


K.set_image_dim_ordering("tf")
CUBE_SIZE = 32
MEAN_PIXEL_VALUE = settings.MEAN_PIXEL_VALUE_NODULE
POS_WEIGHT = 2
NEGS_PER_POS = 20
P_TH = 0.6
LEARN_RATE = 0.001
USE_DROPOUT = True


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
    lw=lw, label='(AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.05])
    plt.ylim([0.0, 1.05])
    # plt.axis('equal')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    #plt.savefig('roc1.png', bbox_inches='tight')
    plt.show()


def get_roc_curve(Y_test_labels, predictions):
    fpr, tpr, thresholds = roc_curve(Y_test_labels[:,1], predictions[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def get_metrics(Y_test_labels, label_predictions):
    cm = confusion_matrix(Y_test_labels[:,1], label_predictions[:,1])

    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    precision = TP*1.0/(TP+FP)
    recall = TP*1.0/(TP+FN)
    specificity = TN*1.0/(TN+FP)

    return precision, recall, specificity, cm


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.show()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    # plt.grid('off')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


def data_generator(batch_size, record_list, train_set=True):
    batch_idx = 0
    means = []
    random_state = numpy.random.RandomState(1301)
    while True:
        img_list = []
        chara_list=[]
        class_list = []
        # if train_set:
            # random.shuffle(record_list)
        CROP_SIZE = CUBE_SIZE
        for record_idx, record_item in enumerate(record_list):
            class_label = record_item[2]
            cube_chara=record_item[1]
            cube_image = record_item[0]
            if train_set:
                if random.randint(0, 100) > 50:
                    cube_image = numpy.fliplr(cube_image)
                if random.randint(0, 100) > 50:
                    cube_image = numpy.flipud(cube_image)
                if random.randint(0, 100) > 50:
                    cube_image = cube_image[:, :, ::-1]
                if random.randint(0, 100) > 50:
                    cube_image = cube_image[:, ::-1, :]
                if random.randint(0, 100) > 50:
                   cube_image = numpy.rot90(cube_image,k=1,axes=(0,1))
                if random.randint(0, 100) > 50:
                    cube_image = numpy.rot90(cube_image,k=2, axes=(0,1))
                if random.randint(0, 100) > 50:
                   cube_image = numpy.rot90(cube_image,k=3,axes=(0,1))
                if random.randint(0, 100) > 50:
                    cube_image = numpy.rot90(cube_image,k=1,axes=(0,2))
                if random.randint(0, 100) > 50:
                    cube_image = numpy.rot90(cube_image,k=2,axes=(0,2))
                if random.randint(0, 100) > 50:
                    cube_image = numpy.rot90(cube_image,k=3,axes=(0,2))
                if random.randint(0, 100) > 50:
                    cube_image = numpy.rot90(cube_image,k=1,axes=(1,2))
                if random.randint(0, 100) > 50:
                    cube_image = numpy.rot90(cube_image,k=2,axes=(1,2))
                if random.randint(0, 100) > 50:
                    cube_image = numpy.rot90(cube_image,k=3,axes=(1,2))
                assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
            means.append(cube_image.mean())
            img3d = prepare_image_for_net3D(cube_image)
            if train_set:
                if len(means) % 1000000 == 0:
                    print("Mean: ", sum(means) / len(means))
            img_list.append(img3d)
            chara_list.append(cube_chara)
            class_list.append(class_label)
            # size_list.append(size_label)

            batch_idx += 1
            if batch_idx >= batch_size:
                x_cube = numpy.vstack(img_list)
                x_chara=numpy.vstack(chara_list)
                y_class = numpy.vstack(class_list)
                # y_size = numpy.vstack(size_list)
                yield [x_cube, x_chara], y_class
                img_list = []
                chara_list=[]
                class_list = []
                # size_list = []
                batch_idx = 0


def get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=None, features=False, mal=False,summary=True) -> Model:
    cube_inputs = Input(shape=(32, 32, 32, 1), name="cube")
    chara_inputs=Input(shape=(9,),name="chara")


    x = cube_inputs
    x = MaxPooling3D(pool_size=(1, 1, 1), strides=(2, 1, 1), padding="same")(x)
    x = Conv3D(64, (3, 3, 3), padding='same', name='conv1', activation="relu",strides=(2, 1, 1))(x)

    x = MaxPooling3D(pool_size=(2, 1, 1), strides=None, padding='valid', data_format=None)(x)
    if True:
        x = Dropout(rate=0.2)(x)


    x = Conv3D(64, (4, 4, 4), padding='same', name='conv2', activation="relu",strides=(2, 1, 1))(x)

    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)
    if True:
        x = Dropout(rate=0.3)(x)


    x = Conv3D(64, (3, 3, 3),  padding='same', name='conv3b', strides=(2, 1, 1))(x)

    x = Activation("relu")(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool3')(x)
    if True:
        x = Dropout(rate=0.5)(x)


    x = Conv3D(64, (1, 1, 1), activation="relu", name="out_class_last")(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool4')(x)
    x = Flatten(name="out_class")(x)

    x = Dense(50, activation="relu")(x)
    concatenated=layers.concatenate([x,chara_inputs],axis=-1)

    x=Dropout(rate=0.3)(concatenated)

    out_class = Dense(2, activation="softmax")(x)
    model = Model(inputs=[cube_inputs,chara_inputs], outputs=out_class)

    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), loss=["binary_crossentropy"], metrics=["acc"])

    if summary:
        model.summary(line_length=140)
    return model


def step_decay(epoch):
    res = 0.001
    if epoch > 5:
        res = 0.0001
    print("learnrate: ", res, " epoch: ", epoch)
    return res


def heat_map_of_9_features(character,y_0):
    chara_mal = numpy.hstack((character, y_0.reshape(684, 1)))
    coe = spearmanr(chara_mal).correlation
    fig, ax = plt.subplots(figsize=(20, 20))
    key_list = ['sphericity', 'margin', 'spiculation', 'texture', 'calcification', 'internalstructure', 'lobulation',
                'subtlety', "diameter", "malignancy"]
    chara_mal_1 = pd.DataFrame(chara_mal, columns=key_list)
    sns.heatmap(pd.DataFrame(coe, columns=key_list, index=key_list), annot=True, vmax=1, vmin=-1, fmt=".3f",
                xticklabels=True,
                yticklabels=True, square=True, cmap="rainbow")
    plt.savefig('xgboost_cube_chara_results/sparman_rainbow_of_9_strutured_feature.png')


def heat_map_of_59_features(activations,y_0):
    mal_59 = numpy.hstack((activations, y_0.reshape(684, 1)))
    coe = spearmanr(mal_59).correlation
    fig, ax = plt.subplots(figsize=(20, 20))
    image_feature_list = ["feature%s" % (i) for i in range(50)]
    key_list = ['sphericity', 'margin', 'spiculation', 'texture', 'calcification', 'internalstructure', 'lobulation',
                'subtlety', "diameter", "malignancy"]
    list_59 = image_feature_list + key_list
    chara_mal_1 = pd.DataFrame(coe, columns=list_59)
    #chara_mal_1.to_csv("59_3.csv", index=False)
    sns.heatmap(pd.DataFrame(coe, columns=list_59, index=list_59), annot=False, vmax=1, vmin=-1,
                xticklabels=True,
                yticklabels=True, square=True, cmap="rainbow")
    plt.savefig('xgboost_cube_chara_results/sparman_rainbow_of_59_features.png')


def single_structured_feature_added_to_50image_features(activations,y_0,plst):
    for k in range(50, 59):
        colums = [s for s in range(50)]
        colums = colums + [k]
        activations_1 = activations[:, colums]
        accs = []
        sens = []
        specs = []
        aucs = []
        tps = []
        tns = []
        fps = []
        fns = []
        i = 1
        for j in range(10):
            print("Cross validation %s" % (j + 1))
            kf = StratifiedKFold(n_splits=5, shuffle=True)
            for train_index, test_index in kf.split(activations_1, y_0):
                dtrain = xgb.DMatrix(activations_1[train_index], y_0[train_index])
                num_rounds = 76
                xgb_model = xgb.train(plst, dtrain, num_rounds)
                dtest = xgb.DMatrix(activations_1[test_index])
                predictions = xgb_model.predict(dtest)
                actuals = y_0[test_index]
                print("Fold", i)
                cm_0 = confusion_matrix(actuals, predictions)
                print(confusion_matrix(actuals, predictions))
                acc_0 = (cm_0[0, 0] + cm_0[1, 1]) / (cm_0[0, 0] + cm_0[0, 1] + cm_0[1, 0] + cm_0[1, 1])
                sen_0 = cm_0[1, 1] / (cm_0[1, 0] + cm_0[1, 1])
                spec_0 = cm_0[0, 0] / (cm_0[0, 0] + cm_0[0, 1])
                rocc = roc_auc_score(actuals, predictions)
                print("acc", acc_0, "  sen", sen_0, "  spec", spec_0, "  auc", rocc)
                tps.append(cm_0[1, 1])
                tns.append(cm_0[0, 0])
                fps.append(cm_0[0, 1])
                fns.append(cm_0[1, 0])
                accs.append(acc_0)
                sens.append(sen_0)
                specs.append(spec_0)
                aucs.append(rocc)
                i += 1
        acc_mean = numpy.mean(numpy.array(accs))
        sen_mean = numpy.mean(numpy.array(sens))
        spec_mean = numpy.mean(numpy.array(specs))
        auc_mean = numpy.mean(numpy.array(aucs))
        metrics_8 = numpy.hstack((numpy.array(accs).reshape(50, 1), numpy.array(sens).reshape(50, 1),
                                  numpy.array(specs).reshape(50, 1), numpy.array(aucs).reshape(50, 1),
                                  numpy.array(tps).reshape(50, 1), numpy.array(tns).reshape(50, 1),
                                  numpy.array(fps).reshape(50, 1), numpy.array(fns).reshape(50, 1)))
        metrics_list = ["acc", "sens", "spec", "auc", "tp", "tn", "fp", "fn"]
        metrics_df = pd.DataFrame(metrics_8, columns=metrics_list)
        metrics_df.to_csv("xgboost_cube_chara_results/single_structured_feature_added_to_50image_features/metrics_of_image50_plus_structure%s.csv" % (k))

        print("10 times 5-fold cross-validation mean of 50 image features added to structured feature %s" % (k), "acc_mean", acc_mean, "  sen_mean", sen_mean, "  spec_mean", spec_mean,
              "  auc_mean", auc_mean)
        print("----------------------------------------------------------------------------------------------------")
    print("done")


def structured_features_added_to_50_image_features_by_relevance(activations,y_0,plst):
    for k in range(10):
        print('Add the first %s most correlative structured features' % (k))
        colums_struture = [58, 52, 56, 54, 57, 51, 53, 50, 55]
        colums = [s for s in range(50)]
        colums = colums + colums_struture[:k]
        activations_1 = activations[:, colums]
        accs = []
        sens = []
        specs = []
        aucs = []
        tps = []
        tns = []
        fps = []
        fns = []
        i = 1
        for j in range(10):
            print("Cross validtion %s" % (j + 1))
            kf = StratifiedKFold(n_splits=5, shuffle=True)
            for train_index, test_index in kf.split(activations_1, y_0):
                dtrain = xgb.DMatrix(activations_1[train_index], y_0[train_index])
                num_rounds = 76
                xgb_model = xgb.train(plst, dtrain, num_rounds)
                dtest = xgb.DMatrix(activations_1[test_index])
                predictions = xgb_model.predict(dtest)
                actuals = y_0[test_index]
                print("Fold", i)
                cm_0 = confusion_matrix(actuals, predictions)
                print(confusion_matrix(actuals, predictions))
                acc_0 = (cm_0[0, 0] + cm_0[1, 1]) / (cm_0[0, 0] + cm_0[0, 1] + cm_0[1, 0] + cm_0[1, 1])
                sen_0 = cm_0[1, 1] / (cm_0[1, 0] + cm_0[1, 1])
                spec_0 = cm_0[0, 0] / (cm_0[0, 0] + cm_0[0, 1])
                rocc = roc_auc_score(actuals, predictions)
                print("acc", acc_0, "  sen", sen_0, "  spec", spec_0, "  auc", rocc)
                tps.append(cm_0[1, 1])
                tns.append(cm_0[0, 0])
                fps.append(cm_0[0, 1])
                fns.append(cm_0[1, 0])
                accs.append(acc_0)
                sens.append(sen_0)
                specs.append(spec_0)
                aucs.append(rocc)
                i += 1
        acc_mean = numpy.mean(numpy.array(accs))
        sen_mean = numpy.mean(numpy.array(sens))
        spec_mean = numpy.mean(numpy.array(specs))
        auc_mean = numpy.mean(numpy.array(aucs))
        metrics_8 = numpy.hstack((numpy.array(accs).reshape(50, 1), numpy.array(sens).reshape(50, 1),
                                  numpy.array(specs).reshape(50, 1), numpy.array(aucs).reshape(50, 1),
                                  numpy.array(tps).reshape(50, 1), numpy.array(tns).reshape(50, 1),
                                  numpy.array(fps).reshape(50, 1), numpy.array(fns).reshape(50, 1)))
        metrics_list = ["acc", "sens", "spec", "auc", "tp", "tn", "fp", "fn"]
        metrics_df = pd.DataFrame(metrics_8, columns=metrics_list)
        metrics_df.to_csv("xgboost_cube_chara_results/structured_features_added_to_50_image_features_by_relevance/metrics_of_image50_plus_most_important%s.csv" % (k))

        print("The average performance of the top %s most important structured feature images  joined with 50 image features" % (k), "acc_mean", acc_mean, "  sen_mean", sen_mean, "  spec_mean", spec_mean,
              "  auc_mean", auc_mean)
        print("----------------------------------------------------------------------------------------------------")
    print("done")


def all_59_features_added_by_relevance(activations,y_0,plst):
    for k in range(59):
        print('Add the first %s most correlative structured or unstructured features' % (k+1))
        colums_struture=[25,21,4,40,28,10,8,3,46,2,30,23,12,26,0,27,58,45,14,31,1,34,35,37,41,48,44,15,33,52,22,9,56,54,57,11,51,38,43,53,50,6,55,47,5,39,17,42,24,13,7,16,18,19,20,29,32,36,49]
        colums=colums_struture[:k+1]
        activations_1 = activations[:,colums]
        accs = []
        sens = []
        specs = []
        aucs = []
        tps = []
        tns = []
        fps = []
        fns = []
        i = 0
        for j in range(10):
            print("Corss validation %s"%(j+1))
            kf = StratifiedKFold(n_splits=5, shuffle=True)
            for train_index, test_index in kf.split(activations_1, y_0):
                dtrain = xgb.DMatrix(activations_1[train_index], y_0[train_index])
                num_rounds = 76
                xgb_model = xgb.train(plst, dtrain, num_rounds)
                dtest = xgb.DMatrix(activations_1[test_index])
                predictions = xgb_model.predict(dtest)
                actuals = y_0[test_index]
                print("Fold",i)
                cm_0=confusion_matrix(actuals, predictions)
                print(confusion_matrix(actuals, predictions))
                acc_0=(cm_0[0,0]+cm_0[1,1])/(cm_0[0,0]+cm_0[0,1]+cm_0[1,0]+cm_0[1,1])
                sen_0=cm_0[1,1]/(cm_0[1,0]+cm_0[1,1])
                spec_0=cm_0[0,0]/(cm_0[0,0]+cm_0[0,1])
                rocc=roc_auc_score(actuals,predictions)
                print("acc",acc_0,"  sen",sen_0,"  spec",spec_0,"  auc",rocc)
                tps.append(cm_0[1,1])
                tns.append(cm_0[0,0])
                fps.append(cm_0[0,1])
                fns.append(cm_0[1, 0])
                accs.append(acc_0)
                sens.append(sen_0)
                specs.append(spec_0)
                aucs.append(rocc)
                i+=1
        acc_mean=numpy.mean(numpy.array(accs))
        sen_mean = numpy.mean(numpy.array(sens))
        spec_mean = numpy.mean(numpy.array(specs))
        auc_mean = numpy.mean(numpy.array(aucs))
        metrics_8=numpy.hstack((numpy.array(accs).reshape(50,1),numpy.array(sens).reshape(50,1),numpy.array(specs).reshape(50,1),numpy.array(aucs).reshape(50,1),numpy.array(tps).reshape(50,1),numpy.array(tns).reshape(50,1),numpy.array(fps).reshape(50,1),numpy.array(fns).reshape(50,1)))
        metrics_list=["acc","sens","spec","auc","tp","tn","fp","fn"]
        metrics_df=pd.DataFrame(metrics_8, columns=metrics_list)
        metrics_df.to_csv("xgboost_cube_chara_results/all_59_features_added_by_relevance/metrics_of_59_most_important%s.csv"%(k+1))


        print("The average performance of the 59 features added by relevance"%(k+1),"acc_mean",acc_mean,"  sen_mean",sen_mean,"  spec_mean",spec_mean,"  auc_mean",auc_mean)
        print("----------------------------------------------------------------------------------------------------")
    print("done")


def train():
    train_files=numpy.load("32\\cubes_12vs45.npy")
    character=numpy.load("32\\chara_12vs45.npy")
    character = numpy.delete(character, 0, axis=1)
    print(train_files.shape[0])
    train_files=train_files.reshape(train_files.shape[0],32,32,32)
    y_0=numpy.load("32\\y_12vs45.npy")
    y=np_utils.to_categorical(y_0,2)

    cube_and_y=zip(train_files,character,y)
    cube_and_y=list(cube_and_y)

    model = get_net(load_weight_path="weight/sudfnn_32/cube_chara_12vs45_32_第{}flod".format(4) + ".h5", summary=False)
    layer_outputs=[layer.output for layer in model.layers[17:18]]
    activation_model=models.Model(inputs=model.input,outputs=layer_outputs)
    data_1= data_generator(len(cube_and_y), cube_and_y, False)
    data=next(data_1)
    activations = activation_model.predict(data[0])

    params = {
        "learning_rate": 0.1,
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 2,
        'gamma': 2,
        'max_depth': 6,
        'lambda': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'silent': 1,
        'eta': 0.1,
        'seed': 1000,
    }

    plst = list(params.items())


    #Plot the correlation heat map of 9 structured features
    heat_map_of_9_features(character, y_0)


    #Plot the correlation heat map of 59 structured and unstructured features
    heat_map_of_59_features(activations, y_0)


    #Calculate the performance of a single structured feature added to 50 image features
    single_structured_feature_added_to_50image_features(activations, y_0, plst)


    #9 structured features are added to 50 image features in order of relevance
    structured_features_added_to_50_image_features_by_relevance(activations, y_0, plst)


    #59 features are added in order of relevance
    all_59_features_added_by_relevance(activations, y_0, plst)

if __name__ == "__main__":
        train()



