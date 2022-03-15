import numpy as np
import sklearn

from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
# record start time
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import random
from matplotlib.pyplot import cm
from keras import layers
# residual block
# https://stackoverflow.com/questions/64792460/how-to-code-a-residual-block-using-two-layers-of-a-basic-cnn-algorithm-built-wit
from tensorflow import keras
import tensorflow as tf

# start timer
start_time = time.time()


class LearningRateReducerCb(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        old_lr = self.model.optimizer.lr.read_value()
        new_lr = old_lr * 0.95
        print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
        self.model.optimizer.lr.assign(new_lr)


input_img = layers.Input(shape=[22, 20, 1], name='img_input')
__next = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1')(input_img)
__next = layers.MaxPooling2D((2, 2))(__next)
__next = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2')(__next)
__next = layers.MaxPooling2D((2, 2))(__next)
__next = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3')(__next)
__next = layers.MaxPooling2D((2, 2))(__next)
__next = layers.Flatten()(__next)
__next = layers.Dense(400, activation='tanh')(__next)
output = layers.Dense(200, activation='softmax', name='type')(__next)
model = keras.Model(inputs=[input_img], outputs=[output])
model.summary()

fileNum = 200
# Get sample data from the dataset
# The feature dimension of the sample is 440 dimensions, the num of sample is 144
sample = np.zeros(0)
tags = np.zeros(0)

for i in range(fileNum):
    filename = '训练集/f' + str(i) + '.dat'
    data = np.fromfile(filename, dtype=np.float32, count=-1, sep='')
    # print(data.shape)

    sample = np.append(sample, data)
    tags = np.append(tags, np.ones(144) * i)

print(tags.shape)
# tags to onehot
tags = keras.utils.to_categorical(tags, num_classes=fileNum)

sample = sample.reshape((-1, 22, 20, 1))

print(sample.shape)
print(tags.shape)
# train test split
x_train, x_dev, y_train, y_dev = sklearn.model_selection.train_test_split(sample, tags, test_size=0.1, random_state=69)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# history = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_dev, y_dev),
#                     callbacks=[early_stopping, LearningRateReducerCb()])
history = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_dev, y_dev),
                    callbacks=[LearningRateReducerCb()])

# plot
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.legend()
plt.show()

# end timer
end_time = time.time()
# save model
model.save('cnn_model.h5')
print('Time cost: ', end_time - start_time)
