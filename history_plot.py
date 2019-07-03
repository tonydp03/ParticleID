import os, sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


file = pd.read_hdf("saved_models/cnn_v1_history.h5", "history").values
plotdir = 'plots/'
val_loss = file[:, 0]
val_accuracy = file[:, 1]
train_loss = file[:, 2]
train_accuracy = file[:, 3]

n_epochs = len(file)
epochs = np.arange(n_epochs)
print("Number of Epochs: ", n_epochs)

print("\nTrain Accuracy: ", train_accuracy)

print("\nTrain Loss: ", train_loss)

print("\nVal Accuracy: ", val_accuracy)

print("\nVal Loss: ", val_loss)


fig1 = plt.figure(1)
plt.plot(epochs, train_accuracy, '-b', label='Training')
plt.plot(epochs, val_accuracy, '-r', label='Validation')

plt.title('Model accuracy', y=1.04)
plt.grid(linestyle=':')
plt.xlabel('Epoch', labelpad=8, fontsize=14)
plt.ylabel('Accuracy', labelpad=10, fontsize=14)
plt.legend(loc='lower right')
plt.savefig(plotdir + 'accuracy.png')
fig1.show()

fig2 = plt.figure(2)
plt.plot(epochs, train_loss, '-b', label='Training')
plt.plot(epochs, val_loss, '-r', label='Validation')

plt.title('Model loss function', y=1.04)
plt.grid(linestyle=':')
plt.xlabel('Epoch', labelpad=8, fontsize=14)
plt.ylabel('Loss', labelpad=10, fontsize=14)
plt.legend(loc='upper right')
plt.savefig(plotdir + 'loss.png')
fig2.show()