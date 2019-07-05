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
import matplotlib.pyplot as plt

from keras import backend as K
from pdb import set_trace

batch_size = 512
channels = 3
img_height = 30
img_width = 50
classes = 4 #5 #6
epochs = 100 #15
dataset_dir = 'data/pkl/'
save_dir = 'saved_models/'
pid_model_name = 'pid'
pid_history_name = 'pid_history'
enreg_model_name = 'enreg'
enreg_history_name = 'enreg_history'

# Load the data
train_dat = pd.read_pickle(dataset_dir + "dataset_train.pkl")
test_dat = pd.read_pickle(dataset_dir + "dataset_test.pkl")

print('Loading training data...')
train_dat = train_dat.query("label!=3")
train_dat = train_dat.query("label!=4")
# slt = train_dat.label>3
slt = train_dat.label==5
# slt2 = (train_dat.label==1) | (train_dat.label==2) 
# train_dat.loc[slt2,"label"] -= 1
train_dat.loc[slt,"label"] -= 2 #3 #1
train_dat.reset_index(drop=True,inplace=True)

x_train, pid_train, en_train = [],[],[]

x_train.append(train_dat.feature)
x_train = np.array(x_train)
x_train = x_train[0]

pid_train.append(train_dat.label)
pid_train = np.array(pid_train)
pid_train = pid_train[0]
pid_train = keras.utils.to_categorical(pid_train, num_classes=classes, dtype='float32')

en_train.append(train_dat.gen_energy)
en_train = np.array(en_train)
en_train = en_train[0]

# for i in range(len(train_dat)):
# #     print('Event number {}'.format(i))
#     x_train.append(train_dat.loc[i].feature)
#     pid_train.append(train_dat.loc[i].label)
#     en_train.append(train_dat.loc[i].gen_energy)
# x_train = np.array(x_train)
# pid_train = np.array(pid_train)
# en_train = np.array(en_train)
# print("Values of pid: ", np.unique(pid_train))
# pid_train = keras.utils.to_categorical(pid_train, num_classes=classes, dtype='float32')
# print('Training data loaded!')

print(x_train.shape)
print(pid_train.shape)
print(en_train.shape)

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
pid_test = keras.utils.to_categorical(pid_test, num_classes=classes, dtype='float32')

en_test.append(test_dat.gen_energy)
en_test = np.array(en_test)
en_test = en_test[0]

# for i in range(len(test_dat)):
#     x_test.append(test_dat.loc[i].feature)
#     pid_test.append(test_dat.loc[i].label)
#     en_test.append(test_dat.loc[i].gen_energy)
# x_test = np.array(x_test)
# pid_test = np.array(pid_test)
# en_test = np.array(en_test)
# print("Values of pid: ", np.unique(pid_test))
# pid_test = keras.utils.to_categorical(pid_test, num_classes=classes, dtype='float32')
# print('Test data loaded!')

print(x_test.shape)
print(pid_test.shape)
print(en_test.shape)


def shower_classification_model():
        input_img = Input(shape=(img_width, img_height, channels), name='input')

        # flat = Flatten()(input_img)
        # dense = Dense(1024, activation='relu', name='dense1')(flat)
        # dense = Dense(128, activation='relu', name='dense2')(dense)
        # dense = Dense(32, activation='relu', name='dense3')(dense)
        
        # conv = Conv2D(3, (1,1), padding='same', activation='relu', data_format='channels_last', name='conv1')(input_img)
        conv = Conv2D(3, (5,1), activation='relu', padding='same', data_format='channels_last', name='conv1')(input_img)
        # pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool1')(conv)

        conv = Conv2D(3, (3,3), activation='relu', padding='same', data_format='channels_last', name='conv2')(conv)
        # pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool2')(conv)

        conv = Conv2D(3, (3,3), activation='relu', padding='same', data_format='channels_last', name='conv3')(conv)
        # pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool3')(conv)

        # conc = Concatenate(axis=3)([conv,conv2])
        # conv = Conv2D(3, (3,3), activation='relu', padding='same', data_format='channels_last', name='conv4')(conv)

        flat = Flatten()(conv)
        # bnorm = BatchNormalization()(flat)

        # dense = Dense(32, activation='relu', name='dense1')(flat)
        # drop = Dropout(0.5)(dense)
        # dense = Dense(32, activation='relu', name='dense2')(drop)
        # drop = Dropout(0.5)(dense)

        pred = Dense(classes, activation='softmax', name='output')(flat)

        model = Model(inputs=input_img, outputs=pred)

        # opt = optimizers.Adam(lr=0.01)
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
        drop = Dropout(0.5)(dense)
        dense = Dense(128, activation='relu', name='dense2')(drop)
        drop = Dropout(0.2)(dense)

        pred = Dense(classes, activation='softmax', name='output')(drop)

        model = Model(inputs=input_img, outputs=pred)

        # opt = optimizers.Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

def pid_and_er_model():
        input_img = Input(shape=(img_width, img_height, channels), name='input')

        conv = Conv2D(3, (3,3), activation='relu', padding='same', data_format='channels_last', name='conv1')(input_img)
        # pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool1')(conv)

        conv = Conv2D(3, (3,3), activation='relu', padding='same', data_format='channels_last', name='conv2')(conv) #(pool)
        # pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool2')(conv)

        conv = Conv2D(3, (3,3), activation='relu', padding='same', data_format='channels_last', name='conv3')(conv) #(pool)
        # pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool3')(conv)
        # conc = Concatenate(axis=3)([conv,conv2])
        # conv = Conv2D(3, (3,3), activation='relu', padding='same', data_format='channels_last', name='conv4')(conv)

        flat = Flatten()(conv)
        # bnorm = BatchNormalization()(flat)

        dense = Dense(1024, activation='relu', name='dense1')(flat)
        # drop = Dropout(0.5)(dense)
        # dense = Dense(32, activation='sigmoid', name='dense2')(drop)
        # drop = Dropout(0.5)(dense)
        pid = Dense(classes, activation='softmax', name='pid_output')(dense)

        conc = Concatenate()([flat,pid])
        bnorm = BatchNormalization()(conc)

        dense = Dense(1024, activation='relu', name='dense2')(bnorm)
        # dense = Dense(128, activation='relu', name='dense3')(dense)        
        # dense = Dense(16, activation='relu', name='dense4')(dense)        
        enreg = Dense(1, name='enreg_output')(dense)

        model = Model(inputs=input_img, outputs=[pid, enreg])

        # opt = optimizers.Adam(lr=0.01)
        model.compile(loss={'pid_output': 'categorical_crossentropy', 'enreg_output': 'mse'}, loss_weights={'pid_output': 1., 'enreg_output': 0.000001}, optimizer='adam', metrics={'pid_output': 'accuracy', 'enreg_output': 'mse'})
        return model

def energy_regression_model():
        # input_img = Input(shape=(img_width, img_height, channels), name='input')
        input_1 = Input(shape=(img_width, img_height, channels), name='input1')
        input_2 = Input(shape=(classes,), name='input2')

        # conv = Conv2D(3, (1,1), padding='same', activation='relu', data_format='channels_last', name='conv1')(input_img)
        conv = Conv2D(3, (3,1), activation='relu', padding='same', data_format='channels_last', name='conv1')(input_1)
        # pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool1')(conv)

        conv = Conv2D(5, (3,3), activation='relu', padding='same', data_format='channels_last', name='conv2')(conv)
        # pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool2')(conv)

        conv = Conv2D(5, (3,3), activation='relu', padding='same', data_format='channels_last', name='conv3')(conv)
        # pool = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool3')(conv)
 
        flat = Flatten()(conv)
        # bnorm = BatchNormalization()(flat)
        conc = Concatenate()([flat,input_2])

        dense = Dense(256, activation='relu', name='dense1')(conc)
        # drop = Dropout(0.5)(dense)
        dense = Dense(64, activation='relu', name='dense2')(dense)
        # drop = Dropout(0.5)(dense)

        enreg = Dense(1, name='enreg_output')(dense)

        model = Model(inputs=[input_1,input_2], outputs=enreg)

        # opt = optimizers.Adam(lr=0.001, decay=1e-3/200)
        model.compile(loss='mse', optimizer='adam') #,metrics=['mse'])
        return model

def energy_regression_model_V2():
        # input_img = Input(shape=(img_width, img_height, channels), name='input')
        input_1 = Input(shape=(img_width*img_height,), name='input1')
        input_2 = Input(shape=(classes,), name='input2')

        # dense = Dense(16, activation='relu', name='dense0')(input_1)
        # dense = Dense(4, activation='relu', name='dense1')(dense)

        conc = Concatenate()([input_1,input_2])        
        bnorm = BatchNormalization()(conc)

        dense = Dense(100, activation='relu', name='dense1')(bnorm)
        dense = Dense(10, activation='relu', name='dense2')(dense)
        # dense = Dense(2, activation='relu', name='dense3')(dense)
        
        enreg = Dense(1, activation='relu', name='enreg_output')(dense)

        model = Model(inputs=[input_1,input_2], outputs=enreg)

        # opt = optimizers.Adam(lr=0.001, decay=1e-3/200)
        model.compile(loss='mse', optimizer='adam') #,metrics=['mse'])
        return model

def energy_regression_model_V3():
        # input_img = Input(shape=(img_width, img_height, channels), name='input')
        input_1 = Input(shape=(1,), name='input1')
        input_2 = Input(shape=(classes,), name='input2')

        conc = Concatenate()([input_1,input_2])
        bnorm = BatchNormalization()(conc)

        dense = Dense(16, activation='relu', name='dense1')(bnorm)
        # drop = Dropout(0.5)(dense)
        dense = Dense(16, activation='relu', name='dense2')(dense)
        # drop = Dropout(0.5)(dense)

        enreg = Dense(1, activation='relu', name='enreg_output')(dense)

        model = Model(inputs=[input_1,input_2], outputs=enreg)

        # opt = optimizers.Adam(lr=0.001, decay=1e-3/200)
        model.compile(loss='mse', optimizer='adam') #,metrics=['mse'])
        return model

print('Creating model...')
model = shower_classification_model()
# model = pid_and_er_model()
model.summary()

history = model.fit(x_train, pid_train, batch_size=batch_size, epochs=10, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)

# history = model.fit(x_train, {'pid_output': pid_train, 'enreg_output': en_train}, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)
history_save = pd.DataFrame(history.history).to_hdf(save_dir + pid_history_name + ".h5", "history", append=False)


# Save model and weights
model.save(save_dir + pid_model_name + ".h5")
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
tf.train.write_graph(frozen_graph, save_dir, pid_model_name + ".pbtxt", as_text=True)
tf.train.write_graph(frozen_graph, save_dir, pid_model_name + ".pb", as_text=False)

print('Model saved')

# Score trained model

scores = model.evaluate(x_test, pid_test, verbose=1)
pid_pred = model.predict(x_train)
# scores = model.evaluate(x_test, {'pid_output': pid_test, 'enreg_output': en_test}, verbose=1)

reco_en_train = x_train[:,:,:,0]
print('Reco En shape: ', reco_en_train.shape)
reco_en_train = reco_en_train.reshape(-1, img_height*img_width)
# reco_en_train = np.sum(reco_en_train, axis=1)
print('Reco En shape: ', reco_en_train.shape)
print('Reco En value: ', reco_en_train)

print('Creating second model...')
# model_2 = energy_regression_model_V2()
model_2 = energy_regression_model_V2()
model_2.summary()

# history_2 = model_2.fit([x_train, pid_pred], en_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)
history_2 = model_2.fit([reco_en_train, pid_pred], en_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)
history_save_2 = pd.DataFrame(history_2.history).to_hdf(save_dir + enreg_history_name + ".h5", "history", append=False)

# Save model and weights
model_2.save(save_dir + enreg_model_name + ".h5")
print('Saved trained model 2 at %s ' % save_dir)

frozen_graph_2 = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model_2.outputs])
tf.train.write_graph(frozen_graph_2, save_dir, enreg_model_name + ".pbtxt", as_text=True)
tf.train.write_graph(frozen_graph_2, save_dir, enreg_model_name + ".pb", as_text=False)

print('Model 2 saved')

reco_en_test = x_test[:,:,:,0]
print('Reco En Test shape: ', reco_en_test.shape)
reco_en_test = reco_en_test.reshape(-1, img_height*img_width)
# reco_en_test = np.sum(reco_en_test, axis=1)
print('Reco En Test shape: ', reco_en_test.shape)
print('Reco En Test value: ', reco_en_test)

pid_results = model.predict(x_test)

energy_results = model_2.predict([reco_en_test, pid_results])
print('*****************')
print('True Particle Energies= {} '.format(en_test))
print('Predicted Particle Energies= {}'.format(energy_results))

# print('True Particle 0 PID= {}'.format(np.argmax(pid_test, axis=1)))
# print('Predicted Particle 0 PID= {}'.format(np.argmax(results[0], axis=1)))
# print('Predicted Particle 0 Energy= {}'.format(results[1]))
# # print('Test loss:', scores[0])
# # print('Test accuracy:', scores[1])
# print('*************')
# print('Scores:', scores)
# mse = (np.square(en_test - results[1])).mean(axis=None)
# print('MSE: ', mse)


print('Done!')
