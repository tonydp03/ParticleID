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

classes = 6
dataset_dir = 'data/pkl/'
save_dir = 'saved_models/'
model_name = 'cnn_v1'
plotdir = 'plots/'

# Load data
test_dat = pd.read_pickle(dataset_dir + "dataset_test.pkl")
print('Loading test data...')
x_test, pid_test, en_test = [],[], []
for i in range(len(test_dat)):
#     print('Event number {}'.format(i))
    x_test.append(test_dat.loc[i].feature)
    pid_test.append(test_dat.loc[i].label)
    en_test.append(test_dat.loc[i].gen_energy)
x_test = np.array(x_test)
pid_test = np.array(pid_test)
en_test = np.array(en_test)
pid_test = keras.utils.to_categorical(pid_test, num_classes=classes, dtype='float32')
print('Test data loaded!')

class_names = np.array(['gamma', 'electron', 'muon', 'tau', 'pion_n', 'pion_c'])

# load model
model = load_model(save_dir + model_name + '.h5')

# score trained model
scores = model.evaluate(x_test, pid_test, verbose=1)
y_pred = model.predict(x_test)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

true = np.argmax(pid_test, axis=1)
pred = np.argmax(y_pred, axis=1)
print('True labels: ', true)
print('Predicted labels: ', pred)

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    
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
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.savefig(plotdir + 'confusion_matrix.png')
plt.show()


print('Done!')
