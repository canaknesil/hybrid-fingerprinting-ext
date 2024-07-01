import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import sys
import util


model_file_prefix = "models/mnist"


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

model = Sequential([
    Flatten(input_shape=(xsize, ysize)),
    Dense(8, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='SGD', # adam has a log of parameters
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

util.save_model(model, f"{model_file_prefix}_{xsize}x{ysize}")



