#!/usr/bin/python
#
from __future__ import print_function
import numpy as np
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#
def load_dataset(filename, nr_features):
    f = open(filename, "r")
    #
    X = list()
    Y = list()
    Z = list()
    for i in f:
        line = i[:-1].split(";")
        #
        x = np.array(line[2:-1]).astype("float64")
        if(len(x) == nr_features):       
            X.append(x)
            Z.append(line[1])
            if(line[0] == 'adenosis'):
            #if(line[0] == 'A'):
            	Y.append(int(0))
            if(line[0] == 'ductal_carcinoma'):
            #if(line[0] == 'DC'):
            	Y.append(int(1))
            if(line[0] == 'fibroadenoma'):
            #if(line[0] == 'F'):
            	Y.append(int(0))
            if(line[0] == 'lobular_carcinoma'):
            #if(line[0] == 'LC'):
            	Y.append(int(1))
            if(line[0] == 'mucinous_carcinoma'):
            #if(line[0] == 'MC'):
               	Y.append(int(1))
            if(line[0] == 'papillary_carcinoma'):
            #if(line[0] == 'PC'):
            	Y.append(int(1))
            if(line[0] == 'phyllodes_tumor'):
            #if(line[0] == 'PT'):
            	Y.append(int(0))
            if(line[0] == 'tubular_adenoma'):
            #if(line[0] == 'TA'):
            	Y.append(int(0))
        else:
            print("Erro: {} {}".format(line[1], len(x)))
    #
    f.close()
    return X, Y, Z
#
#
#
def generate_fold(X, Y, Z, fold_file, zoom):
    imgs_train = list()
    imgs_test = list()
    imgs_val = list()
    f = open(fold_file, "r")
    for i in f:
        linha = i[:-1].split("|")
        if(int(linha[1]) == zoom):
            img = linha[0].split(".")[0]
            if(linha[3] == "train"):
                imgs_train.append(img)
            if(linha[3] == "test"):
                imgs_test.append(img)
            if(linha[3] == "val"):
                imgs_val.append(img)
    f.close()
    X_train = list()
    Y_train = list()
    Z_train = list()
    X_test = list()
    Y_test = list()
    Z_test = list()
    X_val = list()
    Y_val = list()
    Z_val = list()
    #
    for i in range(len(X)):
        tmp_img = Z[i].split("-")
        main_img = tmp_img[0]+"-"+tmp_img[1]+"-"+tmp_img[2]+"-"+tmp_img[3]+"-"+tmp_img[4]
        if(main_img in imgs_train):
            X_train.append(X[i])
            Y_train.append(Y[i])
            Z_train.append(Z[i])
        if(main_img in imgs_test):
            X_test.append(X[i])
            Y_test.append(Y[i])
            Z_test.append(Z[i])
        if(main_img in imgs_val):
            X_val.append(X[i])
            Y_val.append(Y[i])
            Z_val.append(Z[i])
    return X_train, Y_train, Z_train, X_test, Y_test, Z_test, X_val, Y_val, Z_val
#
#
#
X, Y, Z = load_dataset(sys.argv[1], 100)
X_train, Y_train, Z_train, X_test, Y_test, Z_test, X_val, Y_val, Z_val = generate_fold(X, Y, Z, sys.argv[4], int(sys.argv[3]))
del X, Y, Z
#
#
#
out_test = open("folds_tensorflow/"+sys.argv[2]+"-"+sys.argv[3]+"-test.txt" ,"w")
#
for i in range(len(X_test)):
    out_test.write("{}".format(Y_test[i]))
    for j in X_test[i]:
        out_test.write(",{:.6f}".format(j))
    out_test.write("\n")
#
out_test.close()
#
#
#
out_test = open("folds_tensorflow/"+sys.argv[2]+"-"+sys.argv[3]+"-val.txt" ,"w")
#
for i in range(len(X_val)):
    out_test.write("{}".format(Y_val[i]))
    for j in X_val[i]:
        out_test.write(",{:.6f}".format(j))
    out_test.write("\n")
#
out_test.close()
#
#
#
out_test = open("folds_tensorflow/"+sys.argv[2]+"-"+sys.argv[3]+"-train.txt" ,"w")
#
for i in range(len(X_train)):
    out_test.write("{}".format(Y_train[i]))
    for j in X_train[i]:
        out_test.write(",{:.6f}".format(j))
    out_test.write("\n")        
#
out_test.close()
