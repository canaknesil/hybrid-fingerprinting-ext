# MIT License

# Copyright (c) 2024 Can Aknesil

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
#from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import sys
import util
import math
import numpy as np
import idx2numpy as idx


model_file_prefix = sys.argv[1]
emnist_dir = sys.argv[2]

print("model_file_prefix:", model_file_prefix)
print("emnist_dir:", emnist_dir)


# Dataset is devided into independent parts, each part consisting a
# training and testing data, a model being trained for each
# independent part.
n_indep_model = 12

# Number of models trained with different sets of initial weights
# using a single dataset part.
n_init_state = 2

print(f"Training with {n_indep_model} independent datasets, starting from {n_init_state} different initial states.")


# MNIST
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# EMNIST
# Using tensorflow_datasets gave an error.
# Loading from dataset that is manually downloaded from https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
x_train = idx.convert_from_file(emnist_dir + "/emnist-digits-train-images-idx3-ubyte")
y_train = idx.convert_from_file(emnist_dir + "/emnist-digits-train-labels-idx1-ubyte")
x_test = idx.convert_from_file(emnist_dir + "/emnist-digits-test-images-idx3-ubyte")
y_test = idx.convert_from_file(emnist_dir + "/emnist-digits-test-labels-idx1-ubyte")

idx = np.array(range(x_train.shape[0]))
np.random.shuffle(idx)
x_train = x_train[idx,:]
y_train = y_train[idx]
idx = np.array(range(x_test.shape[0]))
np.random.shuffle(idx)
x_test = x_test[idx]
y_test = y_test[idx]

print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)


# Normalize the images to [0, 1] range
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reduce image resolution
xsize = 7
ysize = 7
x_train = tf.image.resize(x_train[..., tf.newaxis], [xsize, ysize]).numpy().squeeze()
x_test = tf.image.resize(x_test[..., tf.newaxis], [xsize, ysize]).numpy().squeeze()

# Convert labels to one-hot encoded vectors
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def split(arr, n, i):
    start = i * (len(arr) // n)
    end = (i + 1) * (len(arr) // n)
    #print(f"dataset from {start} to {end}")
    return arr[start:end]


for i in range(n_indep_model):
    x_train_part = split(x_train, n_indep_model, i)
    y_train_part = split(y_train, n_indep_model, i)
    x_test_part = split(x_test, n_indep_model, i)
    y_test_part = split(y_test, n_indep_model, i)

    data_file = f"{model_file_prefix}_{xsize}x{ysize}_indep-{i}"
    np.save(f"{data_file}_x_train.npy", x_train_part)
    np.save(f"{data_file}_y_train.npy", y_train_part)
    np.save(f"{data_file}_x_test.npy", x_test_part)
    np.save(f"{data_file}_y_test.npy", y_test_part)
    
    for j in range(n_init_state):
        model = Sequential([
            Flatten(input_shape=(xsize, ysize)),
            Dense(8, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.compile(optimizer='SGD', # adam has a log of parameters
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        if i == 0 and j == 0:
            model.summary()

        print(f"Training model {i}-{j}")
        model.fit(x_train_part, y_train_part, epochs=85, batch_size=32, validation_split=0.2)

        test_loss, test_acc = model.evaluate(x_test_part, y_test_part)
        print(f'Test accuracy: {test_acc}')

        model_file = f"{model_file_prefix}_{xsize}x{ysize}_indep-{i}_init-{j}"
        util.save_model(model, model_file)

        
print("Done.")

