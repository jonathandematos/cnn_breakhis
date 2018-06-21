from __future__ import print_function
import datetime
import numpy as np
import sys
from breakhis_generator_validation import TumorToLabel, TumorToLabel8
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#
#
def load_predictions(prediction_file):
    Z = list()
    predict = list()
    label = list()
    #
    with open(prediction_file, "r") as f:
        for i in f:
            # SOB_B_A-14-22549G-100-001.png;1;0.3141;0.4423;
            if( i.find("SOB_") == 0):
                values = i[:-1].split(";")
                Z.append(values[0])
                predict.append(np.array([float(values[2]),float(values[3])]))
                label.append(int(values[1]))
    return Z, label, predict



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
    img_list_dict = list()
    labels_dict = list()
    preds_sum = list()
    preds_vote = list()
    for key,preds in img_dict.items():
        img_list_dict.append(key)
        labels_dict.append(preds[0])
        preds_sum.append(preds[1])
        preds_vote.append(preds[2])
    return img_list_dict, labels_dict, preds_sum, preds_vote
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
def print_prediction(model=None, img_list=None, main_batch_size=None, output_prediction=False, prediction_file=None, patches=False, train_imgs=None):
    #
    preds_proba = list()
    preds = list()
    conf_by_tumor = np.zeros((8,2))
    labels = list()
    class_count = np.array([0,0])
    imgname = list()
    predictions = list()
    #
    #
    if(prediction_file != None and output_prediction == False):
        imgname, labels, predictions = load_predictions(prediction_file)
    else:
        if(output_prediction == True):
            if(prediction_file != None ):
                fpred = open("predictions/"+prediction_file, "r")
            else:
                now = datetime.datetime.now()
                fpred = open("predictions/{}.txt".format(now.strftime("%Y%m%d-%H%M%S")), "r")
        for x, y, z in ReadImgs(img_list, width=WIDTH, height=HEIGHT):
            predictions.append(model.predict(np.array([x])).squeeze())
            labels.append(y.argmax())
            imgname.append(z)
    #
    predictions = np.array(predictions)
    class_count_real, class_count_aug = classes_count(imgname)
    predictions /= class_count_aug
    predictions *= class_count_real
    correct = 0
    total = 0
    print("P_train: ",class_count_aug)
    print("P_real: ",class_count_real)
    #
    for i in range(len(predictions)):   
        if(output_prediction):
            fpred.write("{};{};".format(imgname[i].split("/")[-1], labels[i]))
        class_count[labels[i]] += 1
        preds.append(predictions[i].argmax())
        conf_by_tumor[TumorToLabel8(imgname[i])][np.argmax(predictions[i])] += 1
        if(output_prediction):
            for j in predictions[i]:
                fpred.write("{:.4f};".format(j))
            fpred.write("\n")
    #
    if(output_prediction):
        fpred.close()
    #
    fpr, tpr, _ = roc_curve(labels, predictions[:,0], pos_label=0)
    roc_auc = auc(fpr, tpr)
    print("Test AUC 0: {:.4f}".format(roc_auc))
    #
    fpr, tpr, _ = roc_curve(labels, predictions[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("Test AUC 1: {:.4f}".format(roc_auc))
    #
    plt.ioff()
    fig = plt.figure()
    plt.plot(fpr, tpr, color='red',
            lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.close(fig)
    #
    print(classification_report(labels, preds, target_names=["malign", "benign"]))
    #
    #fpr, tpr, _ = roc_curve(labels, preds_proba, pos_label=1)
    #roc_auc = auc(fpr, tpr)
    #
    #print("Test AUC 1: {:.4f}".format(roc_auc))
    a = confusion_matrix(labels, preds)
    print("Overall accuracy: ",float(a[1][1]+a[0][0])/a.sum())
    print("Confusion matrix:\n",a)
    print("Total elements per class:", class_count)
    if(a[0][0]+a[1][0] != 0 and a[0][1]+a[1][1] != 0):
                print("Accuracy per class: {:.3f} {:.3f} {:.3f}\n".format(float(a[0][0])/(a[0][0]+a[0][1]), float(a[1][1])/(a[1][0]+a[1][1]), (float(a[0][0])/(a[0][0]+a[0][1])+float(a[1][1])/(a[1][0]+a[1][1]))/2))
    else:
                print("Zero predictions in one class.")
    print("Confusion by tumor type: \n", conf_by_tumor)
    if(patches == True):
        img_list_dict, labels_dict, preds_sum, preds_vote = accuracy_by_image(imgname, labels, predictions)
        print("Accuracy by patient sum: ", accuracy_by_patient(img_list_dict, labels_dict, preds_sum))
        print("Accuracy by patient vote: ", accuracy_by_patient(img_list_dict, labels_dict, preds_vote))
    else:
        print("Accuracy by patient: ", accuracy_by_patient(imgname, labels, preds))
    print("Malign  Benign")
    print("M_LC\nM_MC\nM_PC\nM_DC\nB_TA\nB_A\nB_PT\nB_F")
#
#
#
def classes_count(train_imgs):
    class_count_real = np.array([0,0])
    class_count_aug = np.array([0,0])
    for i in train_imgs:
        label = TumorToLabel(i)
        if(i.find("rotat") == -1 and i.find("flip") == -1 and i.find("trans") == -1):
            class_count_real += label
        class_count_aug += label
    return class_count_real.astype("float32")/(class_count_real[0]+class_count_real[1]), class_count_aug.astype("float32")/(class_count_aug[0]+class_count_aug[1])
#
def main():
    print_prediction(prediction_file=sys.argv[1])

if __name__ == "__main__":
    main()
