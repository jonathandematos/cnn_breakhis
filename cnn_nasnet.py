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
import keras.applications.nasnet as nasnet
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
BATCH_SIZE = 16
HOME = os.environ['HOME']
TRAIN_EXPERIMENT = sys.argv[3]
TRAIN_FILE = TRAIN_EXPERIMENT #HOME+"/data/BreaKHis_v1/folds_nonorm_dataaug/dsfold2-100-train.txt"
VAL_FILE = TRAIN_EXPERIMENT.replace("train", "validation") #HOME+"/data/BreaKHis_v1/folds_nonorm_dataaug/dsfold2-100-validation.txt"
TEST_FILE = TRAIN_EXPERIMENT.replace("train", "test") #HOME+"/data/BreaKHis_v1/folds_nonorm_dataaug/dsfold2-100-test.txt"
WIDTH = 331
HEIGHT = 331
#
#
#
def build_cnn():

    vgg_inst = nasnet.NASNetLarge(include_top=True, weights='imagenet', input_tensor=None, input_shape=(HEIGHT,WIDTH,3), pooling=None, classes=1000)

    x = vgg_inst.output
    x = Dense(64, activation="relu")(x)
    x = Dense(2, activation="softmax")(x)
  
    model = Model(inputs=vgg_inst.inputs, outputs=x)

    for i in model.layers:
        i.trainable = True


    trainable_layers = ["input_1","stem_conv1","stem_bn1","activation_1","reduction_conv_1_stem_1",
                            "reduction_bn_1_stem_1","activation_2","activation_4","separable_conv_1_reduction_left",
                            "separable_conv_1_reduction_1_st","separable_conv_1_bn_reduction_l",
                            "separable_conv_1_bn_reduction_1","activation_3","activation_5",
                            "separable_conv_2_reduction_left","separable_conv_2_reduction_1_st",
                            "separable_conv_2_bn_reduction_l","separable_conv_2_bn_reduction_1","activation_6","reduction_add_1_stem_1","separable_conv_1_reduction_righ","activation_8",
                            "activation_10","separable_conv_1_bn_reduction_r","separable_conv_1_reduction_righ","separable_conv_1_reduction_left","activation_7","separable_conv_1_bn_reduction_r",
                            "separable_conv_1_bn_reduction_l","separable_conv_2_reduction_righ","activation_9","activation_11","reduction_left2_stem_1","separable_conv_2_bn_reduction_r",
                            "separable_conv_2_reduction_righ","separable_conv_2_reduction_left","adjust_relu_1_stem_2","reduction_add_2_stem_1","reduction_left3_stem_1","separable_conv_2_bn_reduction_r",
                            "reduction_left4_stem_1","separable_conv_2_bn_reduction_l","reduction_right5_stem_1","zero_padding2d_1","reduction_add3_stem_1","add_1","reduction_add4_stem_1","cropping2d_1",
                            "reduction_concat_stem_1","adjust_avg_pool_1_stem_2","adjust_avg_pool_2_stem_2","activation_12","adjust_conv_1_stem_2","adjust_conv_2_stem_2","reduction_conv_1_stem_2","concatenate_1",
                            "reduction_bn_1_stem_2","adjust_bn_stem_2","activation_13","activation_15","separable_conv_1_reduction_left","separable_conv_1_reduction_1_st","separable_conv_1_bn_reduction_l",
                            "separable_conv_1_bn_reduction_1","activation_14","activation_16","separable_conv_2_reduction_left","separable_conv_2_reduction_1_st","separable_conv_2_bn_reduction_l",
                            "separable_conv_2_bn_reduction_1","activation_17","reduction_add_1_stem_2","separable_conv_1_reduction_righ","activation_19","activation_21","separable_conv_1_bn_reduction_r",
                            "separable_conv_1_reduction_righ","separable_conv_1_reduction_left","activation_18","separable_conv_1_bn_reduction_r","separable_conv_1_bn_reduction_l","separable_conv_2_reduction_righ",
                            "activation_20","activation_22","reduction_left2_stem_2","separable_conv_2_bn_reduction_r","separable_conv_2_reduction_righ","separable_conv_2_reduction_left","adjust_relu_1_0",
                            "reduction_add_2_stem_2","reduction_left3_stem_2","separable_conv_2_bn_reduction_r","reduction_left4_stem_2","separable_conv_2_bn_reduction_l","reduction_right5_stem_2","zero_padding2d_2",
                            "reduction_add3_stem_2","add_2","reduction_add4_stem_2","cropping2d_2","reduction_concat_stem_2","adjust_avg_pool_1_0","adjust_avg_pool_2_0","adjust_conv_1_0","adjust_conv_2_0",
                            "activation_23","concatenate_2","normal_conv_1_0","adjust_bn_0","normal_bn_1_0","activation_24","activation_26","activation_28","activation_30","activation_32","separable_conv_1_normal_left1_0",
                            "separable_conv_1_normal_right1_","separable_conv_1_normal_left2_0","separable_conv_1_normal_right2_","separable_conv_1_normal_left5_0","separable_conv_1_bn_normal_left",
                            "separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","activation_25","activation_27","activation_29",
                            "activation_31","activation_33","separable_conv_2_normal_left1_0","separable_conv_2_normal_right1_","separable_conv_2_normal_left2_0","separable_conv_2_normal_right2_",
                            "separable_conv_2_normal_left5_0","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","normal_left3_0",
                           "normal_left4_0","normal_right4_0","separable_conv_2_bn_normal_left","normal_add_1_0","normal_add_2_0","normal_add_3_0","normal_add_4_0","normal_add_5_0","normal_concat_0",
                            "activation_34","activation_35","adjust_conv_projection_1","normal_conv_1_1","adjust_bn_1","normal_bn_1_1","activation_36","activation_38","activation_40","activation_42","activation_44",
                            "separable_conv_1_normal_left1_1","separable_conv_1_normal_right1_","separable_conv_1_normal_left2_1","separable_conv_1_normal_right2_","separable_conv_1_normal_left5_1",
                            "separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","activation_37",
                            "activation_39","activation_41","activation_43","activation_45","separable_conv_2_normal_left1_1","separable_conv_2_normal_right1_","separable_conv_2_normal_left2_1","separable_conv_2_normal_right2_",
                            "separable_conv_2_normal_left5_1","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ",
                            "normal_left3_1","normal_left4_1","normal_right4_1","separable_conv_2_bn_normal_left"]
#                            "normal_add_1_1","normal_add_2_1","normal_add_3_1","normal_add_4_1","normal_add_5_1",
#                            "normal_concat_1","activation_46","activation_47","adjust_conv_projection_2","normal_conv_1_2","adjust_bn_2","normal_bn_1_2","activation_48","activation_50","activation_52",
#                            "activation_54","activation_56","separable_conv_1_normal_left1_2","separable_conv_1_normal_right1_","separable_conv_1_normal_left2_2","separable_conv_1_normal_right2_",
#                            "separable_conv_1_normal_left5_2","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ",
#                            "separable_conv_1_bn_normal_left","activation_49","activation_51","activation_53","activation_55","activation_57","separable_conv_2_normal_left1_2","separable_conv_2_normal_right1_",
#                            "separable_conv_2_normal_left2_2","separable_conv_2_normal_right2_","separable_conv_2_normal_left5_2","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ",
#                            "separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","normal_left3_2","normal_left4_2","normal_right4_2","separable_conv_2_bn_normal_left","normal_add_1_2",
#                            "normal_add_2_2","normal_add_3_2","normal_add_4_2","normal_add_5_2","normal_concat_2","activation_58","activation_59","adjust_conv_projection_3","normal_conv_1_3","adjust_bn_3",
#                            "normal_bn_1_3","activation_60","activation_62","activation_64","activation_66","activation_68","separable_conv_1_normal_left1_3","separable_conv_1_normal_right1_",
#                            "separable_conv_1_normal_left2_3","separable_conv_1_normal_right2_","separable_conv_1_normal_left5_3","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ",
#                            "separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","activation_61","activation_63","activation_65","activation_67","activation_69",
#                            "separable_conv_2_normal_left1_3","separable_conv_2_normal_right1_","separable_conv_2_normal_left2_3","separable_conv_2_normal_right2_","separable_conv_2_normal_left5_3",
#                            "separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","normal_left3_3","normal_left4_3",
#                            "normal_right4_3","separable_conv_2_bn_normal_left","normal_add_1_3","normal_add_2_3","normal_add_3_3","normal_add_4_3","normal_add_5_3","normal_concat_3","activation_70",
#                            "activation_71","adjust_conv_projection_4","normal_conv_1_4","adjust_bn_4","normal_bn_1_4","activation_72","activation_74","activation_76","activation_78","activation_80",
#                            "separable_conv_1_normal_left1_4","separable_conv_1_normal_right1_","separable_conv_1_normal_left2_4","separable_conv_1_normal_right2_","separable_conv_1_normal_left5_4",
#                            "separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left",
#                            "activation_73","activation_75","activation_77","activation_79","activation_81","separable_conv_2_normal_left1_4","separable_conv_2_normal_right1_","separable_conv_2_normal_left2_4",
#                            "separable_conv_2_normal_right2_","separable_conv_2_normal_left5_4","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left",
#                            "separable_conv_2_bn_normal_righ","normal_left3_4","normal_left4_4","normal_right4_4","separable_conv_2_bn_normal_left","normal_add_1_4","normal_add_2_4","normal_add_3_4",
#                            "normal_add_4_4","normal_add_5_4","normal_concat_4","activation_82","activation_83","adjust_conv_projection_5","normal_conv_1_5","adjust_bn_5","normal_bn_1_5","activation_84",
#                            "activation_86","activation_88","activation_90","activation_92","separable_conv_1_normal_left1_5","separable_conv_1_normal_right1_","separable_conv_1_normal_left2_5",
#                            "separable_conv_1_normal_right2_","separable_conv_1_normal_left5_5","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left",
#                            "separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","activation_85","activation_87","activation_89","activation_91","activation_93","separable_conv_2_normal_left1_5",
#                            "separable_conv_2_normal_right1_","separable_conv_2_normal_left2_5","separable_conv_2_normal_right2_","separable_conv_2_normal_left5_5","separable_conv_2_bn_normal_left",
#                            "separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","normal_left3_5","normal_left4_5","normal_right4_5",
#                            "separable_conv_2_bn_normal_left","normal_add_1_5","normal_add_2_5","normal_add_3_5","normal_add_4_5","normal_add_5_5","normal_concat_5","activation_95","activation_94",
#                            "reduction_conv_1_reduce_6","adjust_conv_projection_reduce_6","reduction_bn_1_reduce_6","adjust_bn_reduce_6","activation_96","activation_98","separable_conv_1_reduction_left",
#                            "separable_conv_1_reduction_1_re","separable_conv_1_bn_reduction_l","separable_conv_1_bn_reduction_1","activation_97","activation_99","separable_conv_2_reduction_left",
#                            "separable_conv_2_reduction_1_re","separable_conv_2_bn_reduction_l","separable_conv_2_bn_reduction_1","activation_100","reduction_add_1_reduce_6","separable_conv_1_reduction_righ",
#                            "activation_102","activation_104","separable_conv_1_bn_reduction_r","separable_conv_1_reduction_righ","separable_conv_1_reduction_left","activation_101","separable_conv_1_bn_reduction_r",
#                            "separable_conv_1_bn_reduction_l","separable_conv_2_reduction_righ","activation_103","activation_105","reduction_left2_reduce_6","separable_conv_2_bn_reduction_r",
#                            "separable_conv_2_reduction_righ","separable_conv_2_reduction_left","adjust_relu_1_7","reduction_add_2_reduce_6","reduction_left3_reduce_6","separable_conv_2_bn_reduction_r",
#                            "reduction_left4_reduce_6","separable_conv_2_bn_reduction_l","reduction_right5_reduce_6","zero_padding2d_3","reduction_add3_reduce_6","add_3","reduction_add4_reduce_6",
#                            "cropping2d_3","reduction_concat_reduce_6","adjust_avg_pool_1_7","adjust_avg_pool_2_7","adjust_conv_1_7","adjust_conv_2_7","activation_106","concatenate_3","normal_conv_1_7",
#                            "adjust_bn_7","normal_bn_1_7","activation_107","activation_109","activation_111","activation_113","activation_115","separable_conv_1_normal_left1_7","separable_conv_1_normal_right1_",
#                            "separable_conv_1_normal_left2_7","separable_conv_1_normal_right2_","separable_conv_1_normal_left5_7","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ",
#                            "separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","activation_108","activation_110","activation_112","activation_114",
#                            "activation_116","separable_conv_2_normal_left1_7","separable_conv_2_normal_right1_","separable_conv_2_normal_left2_7","separable_conv_2_normal_right2_","separable_conv_2_normal_left5_7",
#                            "separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","normal_left3_7","normal_left4_7","normal_right4_7",
#                            "separable_conv_2_bn_normal_left","normal_add_1_7","normal_add_2_7","normal_add_3_7","normal_add_4_7","normal_add_5_7","normal_concat_7","activation_117","activation_118","adjust_conv_projection_8",
#                            "normal_conv_1_8","adjust_bn_8","normal_bn_1_8","activation_119","activation_121","activation_123","activation_125","activation_127","separable_conv_1_normal_left1_8",
#                            "separable_conv_1_normal_right1_","separable_conv_1_normal_left2_8","separable_conv_1_normal_right2_","separable_conv_1_normal_left5_8","separable_conv_1_bn_normal_left",
#                            "separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","activation_120","activation_122",
#                            "activation_124","activation_126","activation_128","separable_conv_2_normal_left1_8","separable_conv_2_normal_right1_","separable_conv_2_normal_left2_8","separable_conv_2_normal_right2_",
#                            "separable_conv_2_normal_left5_8","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","normal_left3_8",
#                            "normal_left4_8","normal_right4_8","separable_conv_2_bn_normal_left","normal_add_1_8","normal_add_2_8","normal_add_3_8","normal_add_4_8","normal_add_5_8","normal_concat_8","activation_129",
#                            "activation_130","adjust_conv_projection_9","normal_conv_1_9","adjust_bn_9","normal_bn_1_9","activation_131","activation_133","activation_135","activation_137","activation_139",
#                            "separable_conv_1_normal_left1_9","separable_conv_1_normal_right1_","separable_conv_1_normal_left2_9","separable_conv_1_normal_right2_","separable_conv_1_normal_left5_9",
#                            "separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left",
#                            "activation_132","activation_134","activation_136","activation_138","activation_140","separable_conv_2_normal_left1_9","separable_conv_2_normal_right1_","separable_conv_2_normal_left2_9",
#                            "separable_conv_2_normal_right2_","separable_conv_2_normal_left5_9","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left",
#                            "separable_conv_2_bn_normal_righ","normal_left3_9","normal_left4_9","normal_right4_9","separable_conv_2_bn_normal_left","normal_add_1_9","normal_add_2_9","normal_add_3_9",
#                            "normal_add_4_9","normal_add_5_9","normal_concat_9","activation_141","activation_142","adjust_conv_projection_10","normal_conv_1_10","adjust_bn_10","normal_bn_1_10",
#                            "activation_143","activation_145","activation_147","activation_149","activation_151","separable_conv_1_normal_left1_1","separable_conv_1_normal_right1_","separable_conv_1_normal_left2_1",
#                            "separable_conv_1_normal_right2_","separable_conv_1_normal_left5_1","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left",
#                            "separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","activation_144","activation_146","activation_148","activation_150","activation_152","separable_conv_2_normal_left1_1",
#                            "separable_conv_2_normal_right1_","separable_conv_2_normal_left2_1","separable_conv_2_normal_right2_","separable_conv_2_normal_left5_1","separable_conv_2_bn_normal_left",
#                            "separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","normal_left3_10","normal_left4_10","normal_right4_10","separable_conv_2_bn_normal_left",
#                            "normal_add_1_10","normal_add_2_10","normal_add_3_10","normal_add_4_10","normal_add_5_10","normal_concat_10","activation_153","activation_154","adjust_conv_projection_11",
#                            "normal_conv_1_11","adjust_bn_11","normal_bn_1_11","activation_155","activation_157","activation_159","activation_161","activation_163","separable_conv_1_normal_left1_1",
#                            "separable_conv_1_normal_right1_","separable_conv_1_normal_left2_1","separable_conv_1_normal_right2_","separable_conv_1_normal_left5_1","separable_conv_1_bn_normal_left",
#                            "separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","activation_156","activation_158",
#                            "activation_160","activation_162","activation_164","separable_conv_2_normal_left1_1","separable_conv_2_normal_right1_","separable_conv_2_normal_left2_1","separable_conv_2_normal_right2_",
#                            "separable_conv_2_normal_left5_1","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ",
#                            "normal_left3_11","normal_left4_11","normal_right4_11","separable_conv_2_bn_normal_left","normal_add_1_11","normal_add_2_11","normal_add_3_11","normal_add_4_11","normal_add_5_11",
#                            "normal_concat_11","activation_165","activation_166","adjust_conv_projection_12","normal_conv_1_12","adjust_bn_12","normal_bn_1_12","activation_167","activation_169","activation_171",
#                            "activation_173","activation_175","separable_conv_1_normal_left1_1","separable_conv_1_normal_right1_","separable_conv_1_normal_left2_1","separable_conv_1_normal_right2_",
#                            "separable_conv_1_normal_left5_1","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ",
#                            "separable_conv_1_bn_normal_left","activation_168","activation_170","activation_172","activation_174","activation_176","separable_conv_2_normal_left1_1","separable_conv_2_normal_right1_",
#                            "separable_conv_2_normal_left2_1","separable_conv_2_normal_right2_","separable_conv_2_normal_left5_1","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ",
#                            "separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","normal_left3_12","normal_left4_12","normal_right4_12","separable_conv_2_bn_normal_left","normal_add_1_12",
#                            "normal_add_2_12","normal_add_3_12","normal_add_4_12","normal_add_5_12","normal_concat_12","activation_178","activation_177","reduction_conv_1_reduce_12","adjust_conv_projection_reduce_1",
#                            "reduction_bn_1_reduce_12","adjust_bn_reduce_12","activation_179","activation_181","separable_conv_1_reduction_left","separable_conv_1_reduction_1_re","separable_conv_1_bn_reduction_l",
#                            "separable_conv_1_bn_reduction_1","activation_180","activation_182","separable_conv_2_reduction_left","separable_conv_2_reduction_1_re","separable_conv_2_bn_reduction_l",
#                            "separable_conv_2_bn_reduction_1","activation_183","reduction_add_1_reduce_12","separable_conv_1_reduction_righ","activation_185","activation_187","separable_conv_1_bn_reduction_r",
#                            "separable_conv_1_reduction_righ","separable_conv_1_reduction_left","activation_184","separable_conv_1_bn_reduction_r","separable_conv_1_bn_reduction_l","separable_conv_2_reduction_righ",
#                            "activation_186","activation_188","reduction_left2_reduce_12","separable_conv_2_bn_reduction_r","separable_conv_2_reduction_righ","separable_conv_2_reduction_left",
#                            "adjust_relu_1_13","reduction_add_2_reduce_12","reduction_left3_reduce_12","separable_conv_2_bn_reduction_r","reduction_left4_reduce_12","separable_conv_2_bn_reduction_l",
#                            "reduction_right5_reduce_12","zero_padding2d_4","reduction_add3_reduce_12","add_4","reduction_add4_reduce_12","cropping2d_4","reduction_concat_reduce_12","adjust_avg_pool_1_13",
#                            "adjust_avg_pool_2_13","adjust_conv_1_13","adjust_conv_2_13","activation_189","concatenate_4","normal_conv_1_13","adjust_bn_13","normal_bn_1_13","activation_190","activation_192",
#                            "activation_194","activation_196","activation_198","separable_conv_1_normal_left1_1","separable_conv_1_normal_right1_","separable_conv_1_normal_left2_1","separable_conv_1_normal_right2_",
#                            "separable_conv_1_normal_left5_1","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ",
#                            "separable_conv_1_bn_normal_left","activation_191","activation_193","activation_195","activation_197","activation_199","separable_conv_2_normal_left1_1","separable_conv_2_normal_right1_",
#                            "separable_conv_2_normal_left2_1","separable_conv_2_normal_right2_","separable_conv_2_normal_left5_1","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ",
#                            "separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","normal_left3_13","normal_left4_13","normal_right4_13","separable_conv_2_bn_normal_left","normal_add_1_13",
#                            "normal_add_2_13","normal_add_3_13","normal_add_4_13","normal_add_5_13","normal_concat_13","activation_200","activation_201","adjust_conv_projection_14","normal_conv_1_14",
#                            "adjust_bn_14","normal_bn_1_14","activation_202","activation_204","activation_206","activation_208","activation_210","separable_conv_1_normal_left1_1","separable_conv_1_normal_right1_",
#                            "separable_conv_1_normal_left2_1","separable_conv_1_normal_right2_","separable_conv_1_normal_left5_1","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ",
#                            "separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","activation_203","activation_205","activation_207","activation_209",
#                            "activation_211","separable_conv_2_normal_left1_1","separable_conv_2_normal_right1_","separable_conv_2_normal_left2_1","separable_conv_2_normal_right2_","separable_conv_2_normal_left5_1",
#                            "separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","normal_left3_14","normal_left4_14",
#                            "normal_right4_14","separable_conv_2_bn_normal_left","normal_add_1_14","normal_add_2_14","normal_add_3_14","normal_add_4_14","normal_add_5_14","normal_concat_14","activation_212",
#                            "activation_213","adjust_conv_projection_15","normal_conv_1_15","adjust_bn_15","normal_bn_1_15","activation_214","activation_216","activation_218","activation_220","activation_222",
#                            "separable_conv_1_normal_left1_1","separable_conv_1_normal_right1_","separable_conv_1_normal_left2_1","separable_conv_1_normal_right2_","separable_conv_1_normal_left5_1",
#                            "separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left",
#                            "activation_215","activation_217","activation_219","activation_221","activation_223","separable_conv_2_normal_left1_1","separable_conv_2_normal_right1_","separable_conv_2_normal_left2_1",
#                            "separable_conv_2_normal_right2_","separable_conv_2_normal_left5_1","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left",
#                            "separable_conv_2_bn_normal_righ","normal_left3_15","normal_left4_15","normal_right4_15","separable_conv_2_bn_normal_left","normal_add_1_15","normal_add_2_15","normal_add_3_15",
#                            "normal_add_4_15","normal_add_5_15","normal_concat_15","activation_224","activation_225","adjust_conv_projection_16","normal_conv_1_16","adjust_bn_16","normal_bn_1_16",
#                            "activation_226","activation_228","activation_230","activation_232","activation_234","separable_conv_1_normal_left1_1","separable_conv_1_normal_right1_","separable_conv_1_normal_left2_1",
#                            "separable_conv_1_normal_right2_","separable_conv_1_normal_left5_1","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left",
#                            "separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","activation_227","activation_229","activation_231","activation_233","activation_235","separable_conv_2_normal_left1_1",
#                            "separable_conv_2_normal_right1_","separable_conv_2_normal_left2_1","separable_conv_2_normal_right2_","separable_conv_2_normal_left5_1","separable_conv_2_bn_normal_left",
#                            "separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","normal_left3_16","normal_left4_16","normal_right4_16","separable_conv_2_bn_normal_left",
#                            "normal_add_1_16","normal_add_2_16","normal_add_3_16","normal_add_4_16","normal_add_5_16","normal_concat_16","activation_236","activation_237","adjust_conv_projection_17",
#                            "normal_conv_1_17","adjust_bn_17","normal_bn_1_17","activation_238","activation_240","activation_242","activation_244","activation_246","separable_conv_1_normal_left1_1",
#                            "separable_conv_1_normal_right1_","separable_conv_1_normal_left2_1","separable_conv_1_normal_right2_","separable_conv_1_normal_left5_1","separable_conv_1_bn_normal_left",
#                            "separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","activation_239","activation_241",
#                            "activation_243","activation_245","activation_247","separable_conv_2_normal_left1_1","separable_conv_2_normal_right1_","separable_conv_2_normal_left2_1","separable_conv_2_normal_right2_",
#                            "separable_conv_2_normal_left5_1","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","normal_left3_17",
#                            "normal_left4_17","normal_right4_17","separable_conv_2_bn_normal_left","normal_add_1_17","normal_add_2_17","normal_add_3_17","normal_add_4_17","normal_add_5_17","normal_concat_17",
#                            "activation_248","activation_249","adjust_conv_projection_18","normal_conv_1_18","adjust_bn_18","normal_bn_1_18","activation_250","activation_252","activation_254","activation_256",
#                            "activation_258","separable_conv_1_normal_left1_1","separable_conv_1_normal_right1_","separable_conv_1_normal_left2_1","separable_conv_1_normal_right2_","separable_conv_1_normal_left5_1",
#                            "separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","separable_conv_1_bn_normal_righ","separable_conv_1_bn_normal_left","activation_251",
#                            "activation_253","activation_255","activation_257","activation_259","separable_conv_2_normal_left1_1","separable_conv_2_normal_right1_","separable_conv_2_normal_left2_1",
#                            "separable_conv_2_normal_right2_","separable_conv_2_normal_left5_1","separable_conv_2_bn_normal_left","separable_conv_2_bn_normal_righ","separable_conv_2_bn_normal_left",
#                            "separable_conv_2_bn_normal_righ","normal_left3_18","normal_left4_18","normal_right4_18","separable_conv_2_bn_normal_left","normal_add_1_18","normal_add_2_18","normal_add_3_18",
#                            "normal_add_4_18","normal_add_5_18","normal_concat_18","activation_260","global_average_pooling2d_1","predictions","dense_1","dense_2"]
#

    #trainable_layers = ["block3_pool","block4_conv1","block4_conv2","block4_conv3","block4_conv4","block4_pool","block5_conv1","block5_conv2","block5_conv3","block5_conv4","block5_pool","flatten","fc1","fc2","predictions","dense_1","dense_2"]

    for i in trainable_layers:
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
