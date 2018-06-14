from __future__ import print_function
import keras
from keras import applications
from keras import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.optimizers import SGD
from breakhis_generator_validation import LoadBreakhisList, Generator, GeneratorImgs, ReadImgs, TumorToLabel
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau, TensorBoard, EarlyStopping
import random
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from keras.models import load_model
from keras import regularizers
import sys
import keras.applications.vgg19
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

import keras.backend.tensorflow_backend as tf_bkend

tf_bkend.set_session(sess)
#
#
#
sys.stdout = open(sys.argv[2], "w")
EPOCHS = 40
BATCH_SIZE = 32
HOME = os.environ['HOME']
TRAIN_EXPERIMENT = sys.argv[3]
TRAIN_FILE = TRAIN_EXPERIMENT #HOME+"/data/BreaKHis_v1/folds_nonorm_dataaug/dsfold2-100-train.txt"
VAL_FILE = TRAIN_EXPERIMENT.replace("train", "validation") #HOME+"/data/BreaKHis_v1/folds_nonorm_dataaug/dsfold2-100-validation.txt"
TEST_FILE = TRAIN_EXPERIMENT.replace("train", "test") #HOME+"/data/BreaKHis_v1/folds_nonorm_dataaug/dsfold2-100-test.txt"
WIDTH = 224
HEIGHT = 224
#
#
#
def build_cnn():

    vgg_inst = vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)

    x = vgg_inst.output
    x = Dense(64, activation="relu")(x)
    x = Dense(2, activation="softmax")(x)
  
    model = Model(inputs=vgg_inst.inputs, outputs=x

    for i in model.layers:
        i.trainable = False

    trainable_layers = [""]

    for i in trainable_layers:
        model.get_layer(i).trainable = True

    sgd = SGD(lr=1e-6, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
#    
#
#
#
#
#    
def set_callbacks(run_name):
    callbacks = list()
    checkpoint = ModelCheckpoint(filepath="models/{}".format(run_name),
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
    
    callbacks.append(checkpoint)
    board = TensorBoard(log_dir='all_logs/{}'.format(run_name), histogram_freq=False,
                            write_graph=False, write_grads=True,
                            write_images=False, embeddings_freq=0,
                            embeddings_layer_names=None, embeddings_metadata=None)
    callbacks.append(board)
    #
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-11)
    callbacks.append(reduce_lr)
    #
    earlyStopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
    callbacks.append(earlyStopping)
    return callbacks
#
#
#
def accuracy_by_image(img_list, labels, predictions):
    img_dict = dict()
    #
    # /home/AP43160/data/BreaKHis_v1/patches/adenosis/SOB_B_A-14-22549G-100-001_0_0.png
    #
    # img_dict (
    #   [[],[soma_preds_classes],[soma_votos_classes]]
    # )
    #
    for i in range(len(img_list)):
        img = img_list[i].split("/")[-1].split("_")[2]
        if(img in img_dict):
            img_dict[img][1] += predictions[i]
            img_dict[img][2][np.argmax(predictions[i])] += 1
        else:
            img_dict[img] = list()
            img_dict[img].append(labels[i])
            img_dict[img].append(predictions[i])
            img_dict[img].append([0,0])
            img_dict[img][2][np.argmax(predictions[i])] += 1
    correct_vote = 0
    correct_sum = 0
    for i in img_dict.keys():
        if(img_dict[i][0] == np.argmax(img_dict[i][1])):
            correct_sum += 1
        if(img_dict[i][0] == np.argmax(img_dict[i][2])):
            correct_vote += 1
    print("Accuracy image:\nSum: {}     Vote: {}".format(float(correct_sum)/len(img_dict), float(correct_vote)/len(img_dict)))
    return 0
#
#
#
def accuracy_by_patient(img_list, labels, predictions):
    img_dict = dict()
    #
    # /home/AP43160/data/BreaKHis_v1/patches/adenosis/SOB_B_A-14-22549G-100-001_0_0.png
    #
    # img_dict (
    #   [[],[soma_preds_classes],[soma_votos_classes]]
    # )
    #
    for i in range(len(img_list)):
        img_strs = img_list[i].split("/")[-1].split("-")
        img = img_strs[0]+img_strs[2]
        if(img in img_dict):
            if(labels[i] == np.argmax(predictions[i])):
                img_dict[img][0] += 1
            img_dict[img][1] += 1
        else:
            img_dict[img] = [0,0]
            if(labels[i] == np.argmax(predictions[i])):
                img_dict[img][0] += 1
            img_dict[img][1] += 1
    total = 0
    patient = 0
    for i in img_dict.keys():
        total += 1
        patient += float(img_dict[i][0])/img_dict[i][1]
    return patient/total
#
#
#
def print_prediction(model, img_list, main_batch_size, run_name, output_prediction=False):
        #
    scores = model.evaluate_generator(GeneratorImgs(img_list, batch_size=main_batch_size, width=WIDTH, height=HEIGHT),
                        steps=len(img_list)/main_batch_size)
        #
    print('Test loss: {:.4f}'.format(scores[0]))
    print('Test accuracy: {:.4f}'.format(scores[1]))
    #
    preds_proba = list()
    preds = list()
    conf_by_tumor = np.zeros((8,2))
    labels = list()
    class_count = np.array([0,0])
    #
    if(output_prediction):
        fpred = open("predictions/{}".format(run_name), "w")
    #
    for x, y, z in ReadImgs(img_list, width=WIDTH, height=HEIGHT):
        predictions = model.predict(np.array([x])).squeeze()
        if(output_prediction):
            fpred.write("{};{};".format(z.split("/")[-1], y.argmax()))
        labels.append(y.argmax())
        class_count += y
        preds.append(predictions.argmax())
        preds_proba.append(predictions[y.argmax()])
        conf_by_tumor[TumorToLabel(z)][np.argmax(predictions)] += 1
        if(output_prediction):
            for j in predictions:
                fpred.write("{:.4f};".format(j))
            fpred.write("\n")
    #
    if(output_prediction):
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
    a = confusion_matrix(labels, preds)
    print("Confusion matrix:\n",a)
    print("Total elements per class:", class_count)
    if(a[0][0]+a[1][0] != 0 and a[0][1]+a[1][1] != 0):
                print("Accuracy per class: {:.3f} {:.3f} {:.3f}\n".format(float(a[0][0])/(a[0][0]+a[0][1]), float(a[1][1])/(a[1][0]+a[1][1]), (float(a[0][0])/(a[0][0]+a[0][1])+float(a[1][1])/(a[1][0]+a[1][1]))/2))
    else:
                print("Zero predictions in one class.")
    print("Confusion by tumor type: ", conf_by_tumor)
    print("Accuracy by patient: ", accuracy_by_patient(img_list, labels, preds))
    print("Malign  Benign")
    print("M_LC\nM_MC\nM_PC\nM_DC\nB_TA\nB_A\nB_PT\nB_F")
#
#
#
def TumorToLabel(tumor):
    tumor_lbl = 8
    if(tumor.find("SOB_M_LC") != -1):
        tumor_lbl = 0
    if(tumor.find("SOB_M_MC") != -1):
        tumor_lbl = 1
    if(tumor.find("SOB_M_PC") != -1):
        tumor_lbl = 2
    if(tumor.find("SOB_M_DC") != -1):
        tumor_lbl = 3
    if(tumor.find("SOB_B_TA") != -1):
        tumor_lbl = 4
    if(tumor.find("SOB_B_A") != -1):
        tumor_lbl = 5
    if(tumor.find("SOB_B_PT") != -1):
        tumor_lbl = 6
    if(tumor.find("SOB_B_F") != -1):
        tumor_lbl = 7
    if(tumor_lbl == 8):
        print(tumor)
    return tumor_lbl
#
#
#
model = build_cnn(int(sys.argv[1]))
model.summary()
print(model.optimizer.get_config())
print("Epochs: ",EPOCHS)
print("Batch-size: ", BATCH_SIZE)
#
#
#
train_imgs = LoadBreakhisList(TRAIN_FILE)
random.shuffle(train_imgs)
val_imgs = LoadBreakhisList(VAL_FILE)
random.shuffle(val_imgs)
#
main_batch_size = BATCH_SIZE
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
model.fit_generator(GeneratorImgs(train_imgs, batch_size=main_batch_size, height=HEIGHT, width=WIDTH), \
        validation_steps=nr_batches_val, \
        validation_data=GeneratorImgs(val_imgs, batch_size=main_batch_size, height=HEIGHT, width=WIDTH), \
        steps_per_epoch=nr_batches, epochs=EPOCHS, verbose=True, max_queue_size=1, \
        workers=1, use_multiprocessing=False, \
        callbacks=set_callbacks("cnn_growing_{}".format(sys.argv[2])))
#
del val_imgs
#
test_imgs = LoadBreakhisList(TEST_FILE)
print("###################################\nTest predictions:")
print_prediction(model, test_imgs, main_batch_size, sys.argv[2], output_prediction=True)
#
print("###################################\nTrain predictions:")
print_prediction(model, train_imgs, main_batch_size, sys.argv[2], output_prediction=False)
#
#
#
exit(0)






test_imgs = LoadBreakhisList(TEST_FILE)  
#random.shuffle(test_imgs)
#test_imgs = test_imgs[:1000]
#
scores = model.evaluate_generator(GeneratorImgs(test_imgs, batch_size=main_batch_size, height=HEIGHT, width=WIDTH), 
        steps=len(test_imgs)/main_batch_size)
#
print('Test loss: {:.4f}'.format(scores[0]))
print('Test accuracy: {:.4f}'.format(scores[1]))
#
preds_proba = list()
preds = list()
labels = list()
preds_list = list()
#
fpred = open("predictions/{}".format(sys.argv[2]), "w")
#
total_class = np.array([0,0])
for x, y, z in ReadImgs(test_imgs):
    predictions = model.predict(np.array([x])).squeeze()
    preds_list.append(predictions)
    total_class += y
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
a = confusion_matrix(labels, preds)
print("Confusion matrix:\n",a)
print("Total for each class:", total_class)
print("Accuracy per class: {:.3f} {:.3f} {:.3f}\n".format(float(a[0][0])/(a[0][0]+a[0][1]), float(a[1][1])/(a[1][0]+a[1][1]), (float(a[0][0])/(a[0][0]+a[0][1])+float(a[1][1])/(a[1][0]+a[1][1]))/2))
#
#accuracy_by_image(test_imgs, labels, preds_list)
#
sys.stdout.close()
exit(0)
