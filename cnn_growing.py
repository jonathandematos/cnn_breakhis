from __future__ import print_function
import keras
from keras import applications
from keras import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.optimizers import SGD
from breakhis_generator_validation import LoadBreakhisList, Generator, GeneratorImgs, ReadImgs
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau, TensorBoard
import random
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from keras.models import load_model
from keras import regularizers
import sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

import keras.backend.tensorflow_backend as tf_bkend

tf_bkend.set_session(sess)
#
#
#
def build_cnn(nr_convs):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), name="conv1", activation='relu', input_shape=(224,224,3), kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    model.add(BatchNormalization(axis=3, name="batch1"))
    model.add(MaxPooling2D(pool_size=(2,2), name="max1"))
    
    if(nr_convs > 1):
        model.add(Conv2D(32, (5, 5), name="conv2", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch2"))
        model.add(AveragePooling2D(pool_size=(2,2), name="max2"))

    if(nr_convs > 2):
        model.add(Conv2D(64, (5, 5), name="conv3", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch3"))
        model.add(MaxPooling2D(pool_size=(2,2), name="max3"))

    if(nr_convs > 3):
        model.add(Conv2D(32, (3, 3), name="conv4", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch4"))
        model.add(MaxPooling2D(pool_size=(2,2), name="max4"))
        
    if(nr_convs > 4):
        model.add(Conv2D(32, (3, 3), name="conv5", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch5"))
        model.add(MaxPooling2D(pool_size=(2,2), name="max5"))

    if(nr_convs > 5):
        model.add(Conv2D(32, (3, 3), name="conv6", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch6"))
        model.add(MaxPooling2D(pool_size=(2,2), name="max6"))

    if(nr_convs > 6):
        model.add(Conv2D(16, (3, 3), name="conv7", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch7"))
        #model.add(MaxPooling2D(pool_size=(2,2), name="max7"))

    if(nr_convs > 7):
        model.add(Conv2D(8, (3, 3), name="conv8", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch8"))
        model.add(MaxPooling2D(pool_size=(2,2), name="max8"))

    if(nr_convs > 8):
        model.add(Conv2D(8, (3, 3), name="conv9", activation='relu'))
        model.add(BatchNormalization(axis=3, name="batch9"))
        model.add(MaxPooling2D(pool_size=(2,2), name="max9"))

    model.add(Dropout(0.25))
    #
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.25))

    model.add(Dense(128, activation='relu'))

    #model.add(Dense(16, activation='relu'))

    model.add(Dense(2, activation='softmax'))
    #
    sgd = SGD(lr=1e-6, decay=4e-5, momentum=0.9, nesterov=True)
    #sgd = SGD(lr=1e-6, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
#    
#
#
model = load_model
#
#
#    
def set_callbacks(run_name):
    callbacks = list()
    checkpoint = ModelCheckpoint(filepath="models/{}".format(run_name),
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
    
    #callbacks.append(checkpoint)
    board = TensorBoard(log_dir='all_logs/cnn_fabio_{}__lr000001_largedense_5x5filter_nesterov_decay00004_20epochs_dataaug_regulconv1'.format(run_name), histogram_freq=0,
                            batch_size=32, write_graph=True, write_grads=False,
                            write_images=False, embeddings_freq=0,
                            embeddings_layer_names=None, embeddings_metadata=None)
    callbacks.append(board)
    #
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-11)
    callbacks.append(reduce_lr)
    #
    return callbacks
#
#
#
model = build_cnn(int(sys.argv[1]))
model.summary()
#
#
#
train_imgs = LoadBreakhisList("folds_nonorm_dataaug/dsfold1-100-train.txt")
val_imgs = LoadBreakhisList("folds_nonorm_dataaug/dsfold1-100-validation.txt")
#
main_batch_size = 64
nr_batches_val = len(val_imgs)/main_batch_size
nr_batches = len(train_imgs)/main_batch_size
#
#
#
#for i,j in  GeneratorImgs(train_imgs, batch_size=main_batch_size):
#    for t in range(len(i)):
#        print(j)
#
#for i,j in  GeneratorImgs(val_imgs, batch_size=main_batch_size):
#    for t in range(len(i)):
#        print(j)
#
#exit(0)
#    
#
#
#


model.fit_generator(GeneratorImgs(train_imgs, batch_size=main_batch_size), \
        validation_steps=nr_batches_val, \
        validation_data=GeneratorImgs(val_imgs, batch_size=main_batch_size), \
        steps_per_epoch=nr_batches, epochs=20, verbose=False, max_queue_size=1, \
        workers=1, use_multiprocessing=False, \
        callbacks=set_callbacks("cnn_growing_{}".format(sys.argv[1])))
#
del train_imgs
del val_imgs
#
test_imgs = LoadBreakhisList("folds_nonorm_dataaug/dsfold1-100-test.txt")  
#
scores = model.evaluate_generator(GeneratorImgs(test_imgs, batch_size=main_batch_size), 
        steps=len(test_imgs)/main_batch_size)
#
print('Test loss: {:.4f}'.format(scores[0]))
print('Test accuracy: {:.4f}'.format(scores[1]))
#
preds_proba = list()
preds = list()
labels = list()
#
fpred = open("preditions-size-{}.txt".format(int(sys.argv[1])), "w")
#
for x, y, z in ReadImgs(test_imgs):
    predictions = model.predict(np.array([x])).squeeze()
    fpred.write("{};{};".format(z.split("/")[-1], y.argmax()))
    labels.append(y.argmax())
    preds.append(predictions.argmax())
    preds_proba.append(predictions[y.argmax()])
    for j in predictions:
        fpred.write("{:.4f};".format(j))
    fpred.write("\n")
#
fpred.close()
#
fpr, tpr, _ = roc_curve(labels, preds_proba, pos_label=0)
roc_auc = auc(fpr, tpr)
#
print("Test AUC 0: {:.4f}".format(roc_auc))
#
print(classification_report(labels, preds, target_names=["malign", "benign"]))
#
#fpr, tpr, _ = roc_curve(labels, preds_proba, pos_label=1)
#roc_auc = auc(fpr, tpr)
#
#print("Test AUC 1: {:.4f}".format(roc_auc))
print("Confusion matrix:\n",confusion_matrix(labels, preds))
#
exit(0)
