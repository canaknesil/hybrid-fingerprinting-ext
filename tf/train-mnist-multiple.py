import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import sys
import util
import math
import numpy as np


model_file_prefix = sys.argv[1]
print("model_file_prefix:", model_file_prefix)


# Dataset is devided into independent parts, each part consisting a
# training and testing data, a model being trained for each
# independent part.
n_indep_model = 2

# Number of models trained with different sets of initial weights
# using a single dataset part.
n_init_state = 2

print(f"Training with {n_indep_model} independent datasets, starting from {n_init_state} different initial states.")


(x_train, y_train), (x_test, y_test) = mnist.load_data()

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
    start = math.floor(i * len(arr) / n)
    end = math.floor((i+1) * len(arr) / n)
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
        model.fit(x_train_part, y_train_part, epochs=20, batch_size=32, validation_split=0.2)

        test_loss, test_acc = model.evaluate(x_test_part, y_test_part)
        print(f'Test accuracy: {test_acc}')

        model_file = f"{model_file_prefix}_{xsize}x{ysize}_indep-{i}_init-{j}"
        util.save_model(model, model_file)

        
print("Done.")

