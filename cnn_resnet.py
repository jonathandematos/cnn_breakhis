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
import keras.applications.resnet50 as resnet50
from process_results import print_prediction
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

import keras.backend.tensorflow_backend as tf_bkend

tf_bkend.set_session(sess)
#
#
#
sys.stdout = open("outputs/"+sys.argv[1], "w")
EPOCHS = 60
BATCH_SIZE = 32
HOME = os.environ['HOME']
TRAIN_EXPERIMENT = sys.argv[2]
TRAIN_FILE = TRAIN_EXPERIMENT #HOME+"/data/BreaKHis_v1/folds_nonorm_dataaug/dsfold2-100-train.txt"
VAL_FILE = TRAIN_EXPERIMENT.replace("train", "validation") #HOME+"/data/BreaKHis_v1/folds_nonorm_dataaug/dsfold2-100-validation.txt"
TEST_FILE = TRAIN_EXPERIMENT.replace("train", "test") #HOME+"/data/BreaKHis_v1/folds_nonorm_dataaug/dsfold2-100-test.txt"
WIDTH = 224
HEIGHT = 224
FIT_VERBOSE = 2
#
#
#
def build_cnn():

    nasnet_inst = resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=(HEIGHT,WIDTH,3), pooling=None, classes=1000)

    x = nasnet_inst.get_layer("flatten_1").output
    x = Dense(1000, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(2, activation="softmax")(x)
  
    model = Model(inputs=nasnet_inst.inputs, outputs=x)

    for i in model.layers:
        i.trainable = True

    non_trainable = ["input_1","conv1_pad","conv1","bn_conv1","activation_1","max_pooling2d_1",
                    "res2a_branch2a","bn2a_branch2a","activation_2",
                    "res2a_branch2b","bn2a_branch2b","activation_3","res2a_branch2c","res2a_branch1","bn2a_branch2c","bn2a_branch1",
                    "add_1","bn2a_branch1","activation_4","res2b_branch2a","bn2b_branch2a","activation_5","res2b_branch2b","bn2b_branch2b","activation_6","res2b_branch2c","bn2b_branch2c",
                    "add_2","activation_4","activation_7","res2c_branch2a","bn2c_branch2a","activation_8","res2c_branch2b","bn2c_branch2b","activation_9","res2c_branch2c","bn2c_branch2c",
                    "add_3","activation_7","activation_10","res3a_branch2a","bn3a_branch2a","activation_11","res3a_branch2b","bn3a_branch2b","activation_12","res3a_branch2c","res3a_branch1","bn3a_branch2c","bn3a_branch1"]
#                    "add_4","bn3a_branch1","activation_13","res3b_branch2a","bn3b_branch2a","activation_14","res3b_branch2b","bn3b_branch2b","activation_15","res3b_branch2c","bn3b_branch2c",
#                    "add_5","activation_13","activation_16","res3c_branch2a","bn3c_branch2a","activation_17","res3c_branch2b","bn3c_branch2b","activation_18","res3c_branch2c","bn3c_branch2c",
#                    "add_6","activation_16","activation_19","res3d_branch2a","bn3d_branch2a","activation_20","res3d_branch2b","bn3d_branch2b","activation_21","res3d_branch2c","bn3d_branch2c",
#                    "add_7","activation_19","activation_22","res4a_branch2a","bn4a_branch2a","activation_23","res4a_branch2b","bn4a_branch2b","activation_24","res4a_branch2c","res4a_branch1","bn4a_branch2c","bn4a_branch1",
#                    "add_8","bn4a_branch1[0][0]","activation_25","res4b_branch2a","bn4b_branch2a","activation_26","res4b_branch2b","bn4b_branch2b","activation_27","res4b_branch2c","bn4b_branch2c",
#                    "add_9","activation_25[0][0]","activation_28","res4c_branch2a","bn4c_branch2a","activation_29","res4c_branch2b","bn4c_branch2b","activation_30","res4c_branch2c","bn4c_branch2c",
#                    "add_10","activation_28[0][0]","activation_31","res4d_branch2a","bn4d_branch2a","activation_32","res4d_branch2b","bn4d_branch2b","activation_33","res4d_branch2c","bn4d_branch2c",
#                    "add_11","activation_31[0][0]","activation_34","res4e_branch2a","bn4e_branch2a","activation_35","res4e_branch2b","bn4e_branch2b","activation_36","res4e_branch2c","bn4e_branch2c",
#                    "add_12","activation_34","activation_37","res4f_branch2a","bn4f_branch2a","activation_38","res4f_branch2b","bn4f_branch2b","activation_39","res4f_branch2c","bn4f_branch2c",
#                    "add_13","activation_37","activation_40","res5a_branch2a","bn5a_branch2a","activation_41","res5a_branch2b","bn5a_branch2b","activation_42","res5a_branch2c","res5a_branch1","bn5a_branch2c","bn5a_branch1",
#                    "add_14","bn5a_branch1","activation_43","res5b_branch2a","bn5b_branch2a","activation_44","res5b_branch2b","bn5b_branch2b","activation_45","res5b_branch2c","bn5b_branch2c",
#                    "add_15","activation_43","activation_46","res5c_branch2a","bn5c_branch2a","activation_47","res5c_branch2b","bn5c_branch2b","activation_48","res5c_branch2c","bn5c_branch2c",
#                    "add_16","activation_46","activation_49","avg_pool","flatten_1","fc1000"]
    for i in non_trainable:
        try:
            model.get_layer(i).trainable = False
        except ValueError:
            print("Layer does not exist: ",i)

    sgd = SGD(lr=1e-6, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
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
model = build_cnn()
model.summary()
#exit(0)
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
        steps_per_epoch=nr_batches, epochs=EPOCHS, verbose=FIT_VERBOSE, max_queue_size=1, \
        workers=1, use_multiprocessing=False, \
        callbacks=set_callbacks(sys.argv[1]))
#
del val_imgs
#
#
#
test_imgs = LoadBreakhisList(TEST_FILE)
fpred = open("predictions/"+sys.argv[1], "w")
predictions = list()
labels = list()
imgname = list()
for x, y, z in ReadImgs(test_imgs, width=WIDTH, height=HEIGHT):
    preds = model.predict(np.array([x])).squeeze()
    predictions.append(preds)
    labels.append(y.argmax())
    imgname.append(z)
    fpred.write("{};{}".format(z.split("/")[-1], y.argmax()))
    for j in preds:
        fpred.write(";{:.4f}".format(j))
    fpred.write("\n")
fpred.close()
#
predictions = np.array(predictions)
#
#
#
print("###################################\nTest predictions:")
print_prediction(patches=True, predictions=predictions, labels=labels, imgname=imgname, prediction_file=None)
#
#print("###################################\nTrain predictions:")
#print_prediction(model=model, img_list=train_imgs, main_batch_size=main_batch_size, output_prediction=True, prediction_file=sys.argv[1], patches=True, train_imgs=train_imgs, WIDTH=WIDTH, HEIGHT=HEIGHT)
