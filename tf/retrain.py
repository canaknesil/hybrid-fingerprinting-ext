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
import sys
import util
import numpy as np


model_prefix = util.remove_trailing_slash(sys.argv[1])    
dataset_prefix = sys.argv[2]
retrained_model_prefix = util.remove_trailing_slash(sys.argv[3])

print("model_prefix:", model_prefix)
print("dataset_prefix:", dataset_prefix)
print("retrained_model_prefix:", retrained_model_prefix)


n_init_state = 2


x_train = np.load(dataset_prefix + "_x_train.npy")
y_train = np.load(dataset_prefix + "_y_train.npy")
x_test = np.load(dataset_prefix + "_x_test.npy")
y_test = np.load(dataset_prefix + "_y_test.npy")

print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)

keras_model = model_prefix + ".keras"
victim_model = tf.keras.models.load_model(keras_model)
print("Victim model summary:")
victim_model.summary()

y_train_from_victim = victim_model.predict(x_train)
print("y_train_from_victim.shape:", y_train_from_victim.shape)

y_train_victim_file = f"{dataset_prefix}_y_train_from_victim.npy"
print(f"Saving y_train_from_victim to {y_train_victim_file}.")
np.save(y_train_victim_file, y_train_from_victim)


def new_model_like(model):
    model_json = model.to_json()
    new_model = tf.keras.models.model_from_json(model_json)
    #model_config = model.get_config()
    #new_model = tf.keras.Model.from_config(model_config)
    return new_model


for i in range(n_init_state):
    model = new_model_like(victim_model)

    model.compile(optimizer='SGD', # adam has a log of parameters
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #model.summary()

    model.fit(x_train, y_train_from_victim, epochs=85, batch_size=32, validation_split=0.2)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')

    test_loss_victim, test_acc_victim = victim_model.evaluate(x_test, y_test)
    print(f'Victim test accuracy: {test_acc_victim}')

    util.save_model(model, f"{retrained_model_prefix}_init-{i}")



