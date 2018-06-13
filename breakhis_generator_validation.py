#!/usr/bin/python
#
from __future__ import print_function
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.utils import np_utils
import numpy
import random
import lmdb
import mahotas.io
import mahotas as mh
from mahotas.features import tas,pftas
#
def LoadBreakhisList(filename):
    #
    f = open(filename,"r")
    file_list = list()
    for i in f:
        file_list.append(i[:-1])
    f.close()
    #
    return file_list
#
def Generator(file_list, batch_size=32, width=224, height=224):
    lmdb_file = "lmdb_test.db/"
    env_handler = lmdb.open(lmdb_file, readonly=True)
    db_handler = env_handler.begin()
    while( 1 ):
        j = 0
        x1 = list()
        x2 = list()
        labels = list()
        for i in file_list:
            if( j >= batch_size ):
                yield [numpy.array(x1), numpy.array(x2)], numpy.array(labels)
                x1 = list()
                x2 = list()
                labels = list()
                j = 0
            img = image.load_img(i, target_size=(height, width))
            x1.append(numpy.array(img_to_array(img)).astype('float32')/255)
            x2.append(numpy.array(ExtractFeature(i, db_handler)))
            y = TumorToLabel(i)
            labels.append(y)
            j += 1
    env_handler.close()
#
def GeneratorImgs(file_list, batch_size=32, width=224, height=224):
    while( 1 ):
        j = 0
        x1 = list()
        labels = list()
        for i in file_list:
            if( j >= batch_size ):
                yield numpy.array(x1), numpy.array(labels)
                x1 = list()
                labels = list()
                img_nms = list()
                j = 0
            img = image.load_img(i, target_size=(height,width))
            x1.append(numpy.array(img_to_array(img)).astype('float32')/255)
            y = TumorToLabel(i)
            labels.append(y)
            j += 1
#
def ReadImgs(file_list, width=224, height=224):
    j = 0
    for i in file_list:
        img = image.load_img(i, target_size=(height,width))
        x1 = numpy.array(img_to_array(img)).astype('float32')/255
        img_nms = i
        labels = TumorToLabel(i)
        yield x1, labels, img_nms
#
def TumorToLabel(tumor):
    if(tumor.find("SOB_B_F") != -1):
        return numpy.array([0,1])
    if(tumor.find("SOB_M_MC") != -1):
        return numpy.array([1,0])
    if(tumor.find("SOB_M_PC") != -1):
        return numpy.array([1,0])
    if(tumor.find("SOB_M_DC") != -1):
        return numpy.array([1,0])
    if(tumor.find("SOB_B_TA") != -1):
        return numpy.array([0,1])
    if(tumor.find("SOB_B_A") != -1):
        return numpy.array([0,1])
    if(tumor.find("SOB_M_LC") != -1):
        return numpy.array([1,0])
    if(tumor.find("SOB_B_PT") != -1):
        return numpy.array([0,1])
    print("Error tumor type: {}".format(tumor))
    return numpy.array([0,1])
#
def ExtractFeature(img_name, db_handler):
    #im = mh.imread(img_name)
    #x = pftas(im)
    a = db_handler.get(img_name.split("/")[-1])
    x = numpy.asfarray(a.split(";"),float)
    return x
