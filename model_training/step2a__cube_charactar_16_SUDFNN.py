import settings
import matplotlib.pyplot as plt
import keras
import random
import numpy
from keras.utils import np_utils
from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, Conv3D,UpSampling2D, merge, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras import backend as K
from itertools import cycle
# limit memory usage..
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import roc_curve, auc, confusion_matrix
import itertools
from keras import layers

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

    # 3rd layer group
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


def train(load_weights_path=None):
    batch_size = 16
    train_files=numpy.load("16\\cubes_12vs45.npy")
    character=numpy.load("16\\chara_12vs45.npy")
    character = numpy.delete(character, 0, axis=1)
    # character = character/ character.max(axis=0)

    print(train_files.shape[0])
    train_files=train_files.reshape(train_files.shape[0],16,16,16)
    y=numpy.load("16\\y_12vs45.npy")

    y=np_utils.to_categorical(y,2)

    cube_and_y=zip(train_files,character,y)
    cube_and_y=list(cube_and_y)

    k=5
    num_validation=len(cube_and_y)//k
    numpy.random.shuffle(cube_and_y)

    validation_scores=[]
    validation_fpr=[]
    validation_tpr=[]
    validation_auc=[]
    validation_precision=[]
    validation_recall=[]
    validation_spec=[]


    for fold in range(k):
        print("Fold{}".format(fold))
        validation_data=cube_and_y[num_validation*fold:num_validation*(fold+1)]
        train_data=cube_and_y[:num_validation*fold]+cube_and_y[num_validation*(fold+1):]
        train_gen = data_generator(batch_size, train_data, True)
        holdout_gen = data_generator(num_validation, validation_data, False)
        print(len(train_data))
        print(len(validation_data))
        model = get_net(load_weight_path=load_weights_path,summary=False)
        callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc', patience=200),
                          keras.callbacks.ModelCheckpoint(filepath="weight/sudfnn_16/cube_chara_12vs45_16_{}flod".format(fold)+".h5", monitor='val_loss',
                                                          save_best_only=True, ),
                          keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, ),

                          # keras.callbacks.CSVLogger("cube_12vs45_{}flod".format(fold)+".csv", separator=',', append=False)
                          ]
        history = model.fit_generator(train_gen, epochs=300, steps_per_epoch=35, validation_data=holdout_gen,
                                      validation_steps=1, callbacks=callbacks_list, shuffle=True, class_weight="auto",
                                      verbose=2)

        nclass = 2
        train_lable=[i[1] for i in train_data]
        val_lable=[i[1] for i in validation_data]
        count=0
        for i in range(nclass):
            for j in val_lable:
                count+=j[i]
            #print("验证集中第{}类样本数".format(i),count)
            count=0
        for i in range(nclass):
            for j in train_lable:
                count+=j[i]
            #print("训练集中第{}类样本数".format(i),count)
            count=0

        model = get_net(load_weight_path="weight/sudfnn_16/cube_chara_12vs45_16_{}flod".format(fold)+".h5",summary=False)
        test_data=next(holdout_gen)
        x=test_data[0]
        y=test_data[1]
        predictions = numpy.vstack(model.predict(x))
        score = model.evaluate_generator(holdout_gen,steps=1)
        label_predictions = numpy.zeros_like(predictions)
        label_predictions[numpy.arange(len(predictions)), predictions.argmax(1)] = 1
        print(predictions.shape, label_predictions.shape)
        fpr, tpr, roc_auc = get_roc_curve(y, predictions)
        precision, recall, specificity, cm = get_metrics(y, label_predictions)


        validation_scores.append(score)
        validation_fpr.append(fpr)
        validation_tpr.append(tpr)
        validation_auc.append(roc_auc)
        validation_precision.append(precision)
        validation_recall.append(recall)
        validation_spec.append(specificity)


        plot_roc_curve(fpr, tpr, roc_auc)
        precision, recall, specificity, cm = get_metrics(y, label_predictions)
        print("precision",precision,"recall", recall, "spec",specificity,"auc",roc_auc)
        # plt.figure()
        plot_confusion_matrix(cm, classes=['low-maligant', 'likely_maligant'], title='Confusion matrix')
        # plt.savefig('confusion_matrix.png', bbox_inches='tight')
        # plt.show()


        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'co', label='Training acc')
        plt.title('Training and validation accuracy of'+" model{}".format(fold))
        plt.plot(epochs, val_acc, 'c', label='Validation acc')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'co', label='Training loss')
        plt.plot(epochs, val_loss, 'c', label='Validation loss')
        plt.title('Training and validation loss of'+" model{}".format(fold))
        plt.legend()
        plt.show()
        print("done")


    print("Metrics of th model:")
    # validation_scores = numpy.average(validation_scores)
    validation_precision=numpy.mean(numpy.array(validation_precision), axis=0)
    validation_loss=numpy.mean(numpy.array(validation_scores), axis=0)[0]
    validation_acc = numpy.mean(numpy.array(validation_scores), axis=0)[1]
    validation_recall=numpy.mean(numpy.array(validation_recall), axis=0)
    validation_spec=numpy.mean(numpy.array(validation_spec), axis=0)
    print("validation_precision",validation_precision)
    print("validation_loss",validation_loss)
    print("validation_acc",validation_acc)
    print("validation_recall",validation_recall)
    print("validation_spec",validation_spec)

    print("multi ROC")
    plt.figure()
    lw = 2
    # plt.plot(validation_fpr[0], validation_tpr[0], color='darkorange',
    #          lw=lw, label='(AUC = %0.2f)' % validation_auc[0])
    # plt.plot(validation_fpr[1], validation_tpr[1], color='aqua',
    #          lw=lw, label='(AUC = %0.2f)' % validation_auc[1])
    # plt.plot(validation_fpr[2], validation_tpr[2], color='seagreen',
    #          lw=lw, label='(AUC = %0.2f)' % validation_auc[2])
    # plt.plot(validation_fpr[3], validation_tpr[3], color='brown',
    #          lw=lw, label='(AUC = %0.2f)' % validation_auc[3])
    # plt.plot(validation_fpr[4], validation_tpr[4], color='blueviolet',
    #          lw=lw, label='(AUC = %0.2f)' % validation_auc[4])
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue',"seagreen","brown"])
    for i, color in zip(range(5), colors):
        plt.plot(validation_fpr[i], validation_tpr[i], color=color, lw=lw,
                 label='ROC curve of model {0} (area = {1:0.3f})'
                       ''.format(i, validation_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.05])
    plt.ylim([0.0, 1.05])
    # plt.axis('equal')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    # plt.savefig('roc1.png', bbox_inches='tight')
    plt.show()




if __name__ == "__main__":

        train(load_weights_path=None)



