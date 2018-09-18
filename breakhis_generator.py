#!/usr/bin/python
#
from __future__ import print_function
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.utils import np_utils
import numpy
import random
import lmdb
import mahotas.io
import mahotas as mh
from mahotas.features import tas,pftas
from keras.preprocessing.image import ImageDataGenerator
#
GRAYSCALE = False
#EXCLUDED = ["SOB_B_A","SOB_B_PT","SOB_B_TA","SOB_M_PC","SOB_M_MC","SOB_M_LC"]
#EXCLUDED = ["SOB_B_A"] #,"SOB_B_PT","SOB_B_TA","SOB_M_PC","SOB_M_MC","SOB_M_LC"]
EXCLUDED = []
#
def LoadBreakhisList(filename):
    #
    f = open(filename,"r")
    file_list = list()
    for i in f:
        addok = True
        for j in EXCLUDED:
            if(i.find(j) != -1):
                addok = False
        if( addok == True ):
            file_list.append(i[:-1])
    f.close()
    #
    return file_list
#
#def Generator(file_list, batch_size=32, width=224, height=224, lmdb_file=None):
#    env_handler = lmdb.open(lmdb_file, readonly=True)
#    db_handler = env_handler.begin()
#    while( 1 ):
#        j = 0
#        x1 = list()
#        x2 = list()
#        labels = list()
#        for i in file_list:
#            if( j >= batch_size ):
#                yield [numpy.array(x1), numpy.array(x2)], numpy.array(labels)
#                x1 = list()
#                x2 = list()
#                labels = list()
#                j = 0
#            img = image.load_img(i, target_size=(height, width), grayscale=GRAYSCALE)
#            x1.append(numpy.array(img_to_array(img)).astype('float32')/255)
#            x2.append(numpy.array(ExtractFeature(i, db_handler)))
#            y = TumorToLabel(i)
#            labels.append(y)
#            j += 1
#    env_handler.close()
#
def GeneratorImgs(file_list, batch_size=32, width=224, height=224, nr_augs=6):
    gen = ImageDataGenerator(rotation_range=45.0, width_shift_range=70.0, height_shift_range=46.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='constant', cval=0.0, horizontal_flip=True, vertical_flip=True)

    while( 1 ):
        j = 0
        x1 = list()
        labels = list()
        for i in file_list:

            if(nr_augs > 0):

                img = image.load_img(i, target_size=(height,width), grayscale=GRAYSCALE)
                img = image.img_to_array(img)
                imgs_flow = gen.flow(numpy.array([img]), batch_size=nr_augs)
             
                for t, new_imgs in enumerate(imgs_flow):
                    x = preprocess_input(new_imgs[0])
                    x1.append(numpy.array(x).astype('float32')/255)
                    y = TumorToLabel(i)
                    labels.append(y)
                    j += 1
    
                    if( j >= batch_size ):
                        yield numpy.array(x1), numpy.array(labels)
                        x1 = list()
                        labels = list()
                        img_nms = list()
                        j = 0
                    if(t >= nr_augs):
                        break
            else:
                img = image.load_img(i, target_size=(height,width), grayscale=GRAYSCALE)
                img = image.img_to_array(img)
             
                x = preprocess_input(img)
                x1.append(numpy.array(x).astype('float32')/255)
                y = TumorToLabel(i)
                labels.append(y)
                j += 1

                if( j >= batch_size ):
                    yield numpy.array(x1), numpy.array(labels)
                    x1 = list()
                    labels = list()
                    img_nms = list()
                    j = 0
               
#
def ReadImgs(file_list, width=224, height=224):
    j = 0
    for i in file_list:
        img = image.load_img(i, target_size=(height,width), grayscale=GRAYSCALE)
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
def TumorToLabel8(tumor):
    tumor_lbl = 8
    if(tumor.find("SOB_M_DC") != -1):
        tumor_lbl = 3
    if(tumor.find("SOB_M_LC") != -1):
        tumor_lbl = 0
    if(tumor.find("SOB_M_MC") != -1):
        tumor_lbl = 1
    if(tumor.find("SOB_M_PC") != -1):
        tumor_lbl = 2
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
def ExtractFeature(img_name, db_handler):
    #im = mh.imread(img_name)
    #x = pftas(im)
    a = db_handler.get(img_name.split("/")[-1])
    x = numpy.asfarray(a.split(";"),float)
    return x
