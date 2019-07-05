"""
Perform inference with Convolutional Neural Network to for ParticleID & Energy Regression in CMS-HGCAL
    
@author: Tony Di Pilato
"""

import itertools
import sys
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd

classes = 4 #5 #6
img_height = 30
img_width = 50
dataset_dir = 'data/pkl/'
save_dir = 'saved_models/'
pid_model_name = 'pid'
enreg_model_name = 'enreg'
plotdir = 'plots/'

# Load data
print('Opening pickle...')
test_dat = pd.read_pickle(dataset_dir + "dataset_test.pkl")
print('Pickle read!')

# x_test =[]
# print('x_test:  ', x_test[2])
# print('x_test[0]:  ', x_test[0])
# print('x_test[0] shape:  ', x_test[0].shape)
# print('shape: ', x_test.shape)
# pid_test = test_dat.label.values
# print('shape: ', pid_test.shape)
# en_test = test_dat.gen_energy.values
# print('shape: ', en_test.shape)

print('Loading test data...')
test_dat = test_dat.query("label!=3") #drop tau
test_dat = test_dat.query("label!=4") #drop pi0
# slt = test_dat.label>3
slt = test_dat.label==5
# slt2 = (test_dat.label==1) | (test_dat.label==2) 
# test_dat.loc[slt2,"label"] -= 1 #merge e-gamma and shift muon label
test_dat.loc[slt,"label"] -= 2 #3 #1 #shift  pi+ labels
test_dat.reset_index(drop=True,inplace=True)

x_test, pid_test, en_test = [],[],[]

x_test.append(test_dat.feature)
x_test = np.array(x_test)
x_test = x_test[0]

pid_test.append(test_dat.label)
pid_test = np.array(pid_test)
pid_test = pid_test[0]

pid_test_label = pid_test
pid_test = keras.utils.to_categorical(pid_test, num_classes=classes, dtype='float32')

en_test.append(test_dat.gen_energy)
en_test = np.array(en_test)
en_test = en_test[0]

gamma_true_en = en_test[pid_test_label==0]
electron_true_en = en_test[pid_test_label==1]
muon_true_en = en_test[pid_test_label==2]
pion_c_true_en = en_test[pid_test_label==3]



# for i in range(len(test_dat)):
#     x_test.append(test_dat.loc[i].feature)
#     pid_test.append(test_dat.loc[i].label)
#     en_test.append(test_dat.loc[i].gen_energy)
# x_test = np.array(x_test)
# pid_test = np.array(pid_test)
# en_test = np.array(en_test)
# pid_test = keras.utils.to_categorical(pid_test, num_classes=classes, dtype='float32')
# print('Test data loaded!')

print(x_test.shape)
print(pid_test.shape)
print(en_test.shape)

#class_names = np.array(['gamma', 'electron', 'muon', 'tau', 'pion_n', 'pion_c'])
# class_names = np.array(['gamma', 'electron', 'muon', 'pion_n', 'pion_c'])
# class_names = np.array(['electromagnetic', 'muon', 'pion_n', 'pion_c'])
class_names = np.array(['gamma', 'electron', 'muon', 'pion_c'])

# load model
model = load_model(save_dir + pid_model_name + '.h5')

# score trained model
scores = model.evaluate(x_test, pid_test, verbose=1)
y_pred = model.predict(x_test)
print('y_pred shape: ', y_pred.shape)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

true = np.argmax(pid_test, axis=1)
pred = np.argmax(y_pred, axis=1)
print('True labels: ', true)
print('Predicted labels: ', pred)
print(np.unique(true))
print(np.unique(pred))

def plot_confusion_matrix(cm, classes, normalize=True, title='Normalized confusion matrix', cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=7)
    plt.yticks(tick_marks, classes, fontsize=7)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=7)

    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.tight_layout()

# Compute confusion matrix
cnf_matrix = confusion_matrix(true, pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(0)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.savefig(plotdir + 'confusion_matrix.png')
plt.show()

model_2 = load_model(save_dir + enreg_model_name + '.h5')
reco_en_test = x_test[:,:,:,0]
reco_en_test = reco_en_test.reshape(-1, img_height*img_width)

en_results = model_2.predict([reco_en_test, y_pred])

gamma_reco_en = en_results[pid_test_label==0]
electron_reco_en = en_results[pid_test_label==1]
muon_reco_en = en_results[pid_test_label==2]
pion_c_reco_en = en_results[pid_test_label==3]

#gamma's energy
fig1 = plt.figure(1)
plt.scatter(gamma_true_en, gamma_reco_en, color='red')
plt.plot([0,0],[450,450], color="black", linestyle='solid')
plt.title('Gamma Energy Regression', y=1.04)
plt.xlabel('True Energy', labelpad=8, fontsize=14)
plt.ylabel('Predicted Energy', labelpad=10, fontsize=14)
plt.savefig(plotdir + 'gammaEn.png')
fig1.show()

#electron's energy
fig2 = plt.figure(2)
plt.scatter(electron_true_en, electron_reco_en, color='blue')
plt.plot([0,0],[450,450], color="black", linestyle='-')
plt.title('Electron Energy Regression', y=1.04)
plt.xlabel('True Energy', labelpad=8, fontsize=14)
plt.ylabel('Predicted Energy', labelpad=10, fontsize=14)
plt.savefig(plotdir + 'electronEn.png')
fig2.show()

#muon's energy
fig3 = plt.figure(3)
plt.scatter(muon_true_en, muon_reco_en, color='green')
plt.plot([0,0],[450,450], color="black", linestyle='-')
plt.title('Muon Energy Regression', y=1.04)
plt.xlabel('True Energy', labelpad=8, fontsize=14)
plt.ylabel('Predicted Energy', labelpad=10, fontsize=14)
plt.savefig(plotdir + 'muonEn.png')
fig1.show()

#pion_c's energy
fig4 = plt.figure(4)
plt.scatter(pion_c_true_en, pion_c_reco_en, color='violet')
plt.plot([0,0],[450,450], color="black", linestyle='-')
plt.title('Charged Pion Energy Regression', y=1.04)
plt.xlabel('True Energy', labelpad=8, fontsize=14)
plt.ylabel('Predicted Energy', labelpad=10, fontsize=14)
plt.savefig(plotdir + 'pion_cEn.png')
fig4.show()

print('Done!')