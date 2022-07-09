import settings
import matplotlib.pyplot as plt
import random
import numpy
from keras.utils import np_utils
from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, Conv3D,UpSampling2D, merge, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras import backend as K
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
import shap
from sklearn import cross_validation, metrics
from sklearn.model_selection import train_test_split,StratifiedKFold,train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error,roc_auc_score


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=config))

K.set_image_dim_ordering("tf")
CUBE_SIZE = 16
MEAN_PIXEL_VALUE = settings.MEAN_PIXEL_VALUE_NODULE
POS_WEIGHT = 2
NEGS_PER_POS = 20
P_TH = 0.6
LEARN_RATE = 0.001
USE_DROPOUT = True


def modelfit(alg, x_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, shuffle=True)
        print(cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])


    # Fit the algorithm on the data
    alg.fit(x_train, y_train, eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(x_train)
    dtrain_predprob = alg.predict_proba(x_train)[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))
    plot_importance(alg, height=0.2, color="cyan")
    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')


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
    # img -= MEAN_PIXEL_VALUE
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
        if train_set:
            random.shuffle(record_list)
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
    cube_inputs = Input(shape=(16, 16, 16, 1), name="cube")
    chara_inputs=Input(shape=(9,),name="chara")


    x = cube_inputs
    x = MaxPooling3D(pool_size=(1, 1, 1), strides=(2, 1, 1), border_mode="same")(x)
    x = Conv3D(64, (3, 3, 3), padding='same', name='conv1', activation="relu",strides=(2, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 1, 1), strides=None, padding='valid', data_format=None)(x)
    if True:
        x = Dropout(rate=0.2)(x)

    # 2nd layer group
    x = Conv3D(128, (4, 4, 4), padding='same', name='conv2', activation="relu",strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)
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


def data_prepare():
    batch_size = 16
    train_files = numpy.load("16\\cubes_12vs45.npy")
    character = numpy.load("16\\chara_12vs45.npy")
    character = numpy.delete(character, 0, axis=1)
    character = character / character.max(axis=0)
    print(train_files.shape[0])
    train_files = train_files.reshape(train_files.shape[0], 16, 16, 16)
    y_0 = numpy.load("16\\y_12vs45.npy")
    y = np_utils.to_categorical(y_0, 2)
    cube_and_y = zip(train_files, character, y)
    cube_and_y = list(cube_and_y)

    model = get_net(load_weight_path="weight/sudfnn_16/cube_chara_12vs45_16_第{}flod".format(2) + ".h5",
                    summary=True)
    layer_outputs = [layer.output for layer in model.layers[13:14]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    data_1 = data_generator(len(cube_and_y), cube_and_y, False)
    data = next(data_1)
    activations = activation_model.predict(data[0])
    y = [numpy.argmax(l) for l in data[1]]
    y = numpy.array(y)
    return activations,y,y_0


def shap_plot(model,X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    cnn_name = ["Feature %s" % (i) for i in range(50)]
    names_9 = ["subtlety", "internalStructure", "calcification", "sphericity", "margin", "lobulation", "spiculation",
               "texture", "diameter"]
    feature_names = cnn_name + names_9
    shap.summary_plot(shap_values[0], X_test, feature_names=feature_names)
    shap.summary_plot(shap_values[0], X_test, plot_type="bar", feature_names=feature_names)


def get_metrics123(y_test,ans):
    cnt1 = 0  #tp+tn
    cnt2 = 0  #fp+fn
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(len(y_test)):
        if y_test[i] == 1 and ans[i] == 1:
            tp+=1
        elif y_test[i] == 1 and ans[i] == 0:
            fn+=1
        elif y_test[i] == 0 and ans[i] == 0:
            tn+=1
        elif y_test[i] == 0 and ans[i] == 1:
            fp+=1
    acc=(tp+tn)/(tp+tn+fp+fn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    specificity=tn/(tn+fp)
    auc_score=roc_auc_score(y_test, ans)
    print("TP:%s,FP:%s,TN:%s,FN:%s"%(tp,fp,tn,fn))
    print("acc:%s,precision:%s,recall:%s,specificity:%s"%(acc,precision,recall,specificity))
    print(auc_score)
    return acc,precision,recall,specificity,auc_score

def roc_plot2(y_test,ans):
    fpr, tpr, threshold = roc_curve(y_test, ans)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()


def train():
    activations,y,y_0=data_prepare()
    acc_list=[]
    recall_list=[]
    spec_list=[]
    precision_list=[]
    auc_score_list=[]

    for k in range(1,11):
        print('Cross_validation %s'%k)
        kf = StratifiedKFold(n_splits=5,shuffle=True)
        i = 1
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

        for train_index, test_index in kf.split(activations, y):
            print("CV_%s_%s_fold" % (k,i))
            X_train=activations[train_index]
            y_train=y[train_index]
            X_test= activations[test_index]
            y_test = y[test_index]

            dtrain = xgb.DMatrix(X_train, y_train)
            num_rounds = 72
            model = xgb.train(plst, dtrain, num_rounds)

            dtest = xgb.DMatrix(X_test)
            ans = model.predict(dtest)

            #shap_plot(model, X_test)
            #roc_plot2(y_test, ans)

            s_45=sum(y_0)
            s_12=len(y_0)-s_45
            s_45_test = sum(y_test)
            s_12_test = len(y_test)-s_45_test
            s_45_train = sum(y_train)
            s_12_train = len(y_train)-s_45_train
            print("Samples_45:%s，Samples_12:%s，Samples_45_in_train_set:%s，Samples_12_in_train_set:%s，Samples_45_in_test_set:%s，Samples_12_in_test_set:%s"%(s_45,s_12,s_45_train,s_12_train,s_45_test,s_12_test))

            acc,precision,recall,specificity,auc_score=get_metrics123(y_test,ans)
            acc_list.append(acc)
            recall_list.append(recall)
            spec_list.append(specificity)
            precision_list.append(precision)
            auc_score_list.append(auc_score)

            i+=1
    acc_mean=numpy.mean(acc_list)
    recall_mean=numpy.mean(recall_list)
    spec_mean=numpy.mean(spec_list)
    prcs_mean=numpy.mean(precision_list)
    auc_mean=numpy.mean(auc_score_list)

    print('ACC_MEAN_10_CV:%0.3f, RECALL_MEAN_10_CV:%0.3f, SPEC_MEAN_10_CV:%0.3f, PRCS_MEAN_10_CV:%0.3f,AUC_MEAN_10_CV:%0.3f'%(acc_mean,recall_mean,spec_mean,prcs_mean,auc_mean,))



def train2():
    activations, y, y_0 = data_prepare()
    activations = activations[:, :50]

    i = 1
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

    acc_list = []
    recall_list = []
    spec_list = []
    precision_list = []
    auc_score_list = []

    for k in range(1, 11):
        print('Cross_validation %s' % k)
        kf = StratifiedKFold(n_splits=5,shuffle=True)
        for train_index, test_index in kf.split(activations, y):
            print("CV_%s_%s_fold" % (k, i))
            X_train = activations[train_index]
            y_train = y[train_index]
            X_test = activations[test_index]
            y_test = y[test_index]

            dtrain = xgb.DMatrix(X_train, y_train)
            num_rounds = 72
            model = xgb.train(plst, dtrain, num_rounds)


            dtest = xgb.DMatrix(X_test)
            ans = model.predict(dtest)

            # shap_plot(model, X_test)
            # roc_plot2(y_test, ans)

            s_45 = sum(y_0)
            s_12 = len(y_0) - s_45
            s_45_test = sum(y_test)
            s_12_test = len(y_test) - s_45_test
            s_45_train = sum(y_train)
            s_12_train = len(y_train) - s_45_train
            print(
                "Samples_45:%s，Samples_12:%s，Samples_45_in_train_set:%s，Samples_12_in_train_set:%s，Samples_45_in_test_set:%s，Samples_12_in_test_set:%s" % (
                s_45, s_12, s_45_train, s_12_train, s_45_test, s_12_test))

            acc, precision, recall, specificity, auc_score = get_metrics123(y_test, ans)
            acc_list.append(acc)
            recall_list.append(recall)
            spec_list.append(specificity)
            precision_list.append(precision)
            auc_score_list.append(auc_score)

            i += 1
        acc_mean = numpy.mean(acc_list)
        recall_mean = numpy.mean(recall_list)
        spec_mean = numpy.mean(spec_list)
        prcs_mean = numpy.mean(precision_list)
        auc_mean = numpy.mean(auc_score_list)

        print(
            'ACC_MEAN_10_CV:%0.3f, RECALL_MEAN_10_CV:%0.3f, SPEC_MEAN_10_CV:%0.3f, PRCS_MEAN_10_CV:%0.3f,AUC_MEAN_10_CV:%0.3f' % (
            acc_mean, recall_mean, spec_mean, prcs_mean, auc_mean,))



if __name__ == "__main__":

        train()

        train2()

