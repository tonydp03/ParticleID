"""
Train a Convolutional Neural Network for ParticleID & Energy Regression in CMS-HGCAL

@author: Tony Di Pilato
"""

import sys
import tensorflow as tf
import keras
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras import optimizers

import os
import numpy as np
import pandas as pd

from keras import backend as K
from pdb import set_trace

batch_size = 256
channels = 3
img_height = 30
img_width = 50
classes = 5 #6
epochs = 15
dataset_dir = 'data/pkl/'
save_dir = 'saved_models/'
model_name = 'cnn_v1'
history_name = 'cnn_v1_history'

# Load the data
train_dat = pd.read_pickle(dataset_dir + "dataset_train.pkl")
test_dat = pd.read_pickle(dataset_dir + "dataset_test.pkl")

print('Loading training data...')
train_dat = train_dat.query("label!=3")
slt = train_dat.label>3
train_dat.loc[slt,"label"] -= 1
train_dat.reset_index(drop=True,inplace=True)

x_train, pid_train, en_train = [],[], []
for i in range(len(train_dat)):
#     print('Event number {}'.format(i))
    x_train.append(train_dat.loc[i].feature)
    pid_train.append(train_dat.loc[i].label)
    en_train.append(train_dat.loc[i].gen_energy)
x_train = np.array(x_train)
pid_train = np.array(pid_train)
en_train = np.array(en_train)
pid_train = keras.utils.to_categorical(pid_train, num_classes=classes, dtype='float32')
print('Training data loaded!')

print(x_train.shape)
print(pid_train.shape)
print(en_train.shape)

print('Loading test data...')
test_dat = test_dat.query("label!=3")
slt = test_dat.label>3
test_dat.loc[slt,"label"] -= 1
test_dat.reset_index(drop=True,inplace=True)

x_test, pid_test, en_test = [],[], []
for i in range(len(test_dat)):
    x_test.append(test_dat.loc[i].feature)
    pid_test.append(test_dat.loc[i].label)
    en_test.append(test_dat.loc[i].gen_energy)
x_test = np.array(x_test)
pid_test = np.array(pid_test)
en_test = np.array(en_test)
pid_test = keras.utils.to_categorical(pid_test, num_classes=classes, dtype='float32')
print('Test data loaded!')

print(x_test.shape)
print(pid_test.shape)
print(en_test.shape)


def shower_classification_model():
        input_img = Input(shape=(img_width, img_height, channels), name='input')

        conv = Conv2D(3, (1,1), padding='same', data_format='channels_last', name='conv1')(input_img)
        conv = Conv2D(16, (5,1), activation='relu', padding='same', data_format='channels_last', name='conv2')(conv)
        conv = Conv2D(32, (3,3), activation='relu', padding='same', data_format='channels_last', name='conv3')(conv)
        conv = Conv2D(64, (4,4), activation='relu', padding='same', data_format='channels_last', name='conv4')(conv)

        flat = Flatten()(conv)
        # bnorm = BatchNormalization()(flat)

        dense = Dense(1024, activation='relu', name='dense1')(flat)
        # drop = Dropout(0.5)(dense)
        dense = Dense(128, activation='relu', name='dense2')(dense)#(drop)
        # drop = Dropout(0.2)(dense)

        pred = Dense(classes, activation='softmax', name='output')(dense)#(drop)

        model = Model(inputs=input_img, outputs=pred)

        opt = optimizers.Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

def shower_classification_model_V2():
        input_img = Input(shape=(img_width, img_height, channels), name='input')

        l1 = Conv2D(3, (1,1), padding='same', data_format='channels_last', name='conv1')(input_img)

        l2 = Conv2D(5, (1,1), activation='relu', padding='same', data_format='channels_last', name='conv2')(input_img)
        l2 = Conv2D(5, (5,1), activation='relu', padding='same', data_format='channels_last', name='conv3')(l2)
        
        l3 = Conv2D(5, (1,1), activation='relu', padding='same', data_format='channels_last', name='conv4')(input_img)
        l3 = Conv2D(5, (3,3), activation='relu', padding='same', data_format='channels_last', name='conv5')(l3)

        l4 = MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='same', data_format='channels_last', name='pool1')(input_img)
        l4 = Conv2D(5, (1,1), activation='relu', padding='same', data_format='channels_last', name='conv6')(l4)

        conc = Concatenate(axis=3)([l1,l2,l3,l4])

        flat = Flatten()(conc)
        # bnorm = BatchNormalization()(flat)

        dense = Dense(1024, activation='relu', name='dense1')(flat)
        # drop = Dropout(0.5)(dense)
        dense = Dense(128, activation='relu', name='dense2')(drop)
        # drop = Dropout(0.2)(dense)

        pred = Dense(classes, activation='softmax', name='output')(drop)

        model = Model(inputs=input_img, outputs=pred)

        opt = optimizers.Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

print('Creating model...')
model = shower_classification_model()
model.summary()

history = model.fit(x_train, pid_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)
history_save = pd.DataFrame(history.history).to_hdf(save_dir + history_name + ".h5", "history", append=False)


# Save model and weights
model.save(save_dir + model_name + ".h5")
print('Saved trained model at %s ' % save_dir)

# save the frozen model
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, save_dir, model_name + ".pbtxt", as_text=True)
tf.train.write_graph(frozen_graph, save_dir, model_name + ".pb", as_text=False)

print('Model saved')

# Score trained model

scores = model.evaluate(x_test, pid_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

print('Done!')
