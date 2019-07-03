"""
Train a Convolutional Neural Network to classify Sentinel-2 images

@author: Tony Di Pilato
"""

import sys
import tensorflow as tf
import keras
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras import optimizers

import os
import numpy as np
import pandas as pd

from keras import backend as K
from pdb import set_trace

batch_size = 1024
channels = 3
img_height = 30
img_width = 50
classes = 6
epochs = 50
dataset_dir = 'data/pkl/'
save_dir = 'saved_models/'
model_name = 'cnn_v1'
history_name = 'cnn_v1_history'

# Load the data
train_dat = pd.read_pickle(dataset_dir + "dataset_train.pkl")
test_dat = pd.read_pickle(dataset_dir + "dataset_test.pkl")

print('Loading training data...')
x_train, pid_train, en_train = [],[], []
for i in range(len(train_dat)):
#     print('Event number {}'.format(i))
    x_train.append(train_dat.loc[i].feature)
    pid_train.append(train_dat.loc[i].label)
    en_train.append(train_dat.loc[i].gen_energy)
x_train = np.array(x_train)
pid_train = np.array(pid_train)
en_train = np.array(en_train)
# x_train = x_train.reshape(-1,channels,img_height,img_width)
pid_train = keras.utils.to_categorical(pid_train, num_classes=classes, dtype='float32')
print('Training data loaded!')

print(x_train.shape)
print(pid_train.shape)
print(en_train.shape)

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
# x_test = x_test.reshape(-1,channels,img_height,img_width)
pid_test = keras.utils.to_categorical(pid_test, num_classes=classes, dtype='float32')
print('Test data loaded!')

print(x_test.shape)
print(pid_test.shape)
print(en_test.shape)

# x_val,y_val = [],[]
# for i in range(len(val_dat)):
#     x_val.append(val_dat.loc[i].feature)
#     y_val.append(val_dat.loc[i].label)
# x_val = np.array(x_val)
# y_val = np.array(y_val)
# x_val = x_val.reshape(-1,channels,52)
# y_val = keras.utils.to_categorical(y_val, num_classes=classes, dtype='float32')

# print(x_val.shape)
# print(y_val.shape)


def shower_classification_model():
    input_img = Input(shape=(img_width, img_height, channels), name='input')
    bnorm = BatchNormalization()(input_img)
    
    conv = Conv2D(16, (5,3), activation='relu', padding='same', data_format='channels_last', name='conv1')(bnorm)
#    bnorm = BatchNormalization()(conv)
#    pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool1')(bnorm)

    conv = Conv2D(16, (4,4), activation='relu', padding='same', data_format='channels_last', name='conv2')(conv)
#    bnorm = BatchNormalization()(conv)
#    pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool2')(bnorm)

    conv = Conv2D(32, (5,5), activation='relu', padding='same', data_format='channels_last', name='conv3')(conv)
#    bnorm = BatchNormalization()(conv)
#    pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool3')(bnorm)

#     bnorm = BatchNormalization()(conv)
#     pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool1')(bnorm)

#     conv = Conv2D(64, (3,3), activation='relu', padding='same', data_format='channels_last', name='conv3')(pool)
#     conv = Conv2D(64, (3,3), activation='relu', padding='same', data_format='channels_last', name='conv4')(conv)
#     bnorm = BatchNormalization()(conv)
#     pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool2')(bnorm)

    flat = Flatten()(conv)

    dense = Dense(32, activation='relu', name='dense1')(flat)
#    dense = Dense(32, activation='sigmoid', name='dense2')(dense)
    pred = Dense(classes, activation='softmax', name='output')(dense)

    model = Model(inputs=input_img, outputs=pred)


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

print('Creating model...')
model = shower_classification_model()
model.summary()

history = model.fit(x_train, pid_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True, verbose=1)
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
