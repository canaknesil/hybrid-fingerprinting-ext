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
import sys
import tensorflow as tf


model_path = sys.argv[1]

# model_path shouldn't end with / because .tflite extension will be
# added to the end.
while model_path[-1] == '/':
    model_path = model_path[:-1]
    
print("model_path:", model_path)

model_path_keras = model_path + ".keras"


# Evaluate model

# Assuming .keras file exists.
model = tf.keras.models.load_model(model_path_keras)
model.summary()
# TODO: Decide whether the model operations are support on TfLite Micro.


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
tflite_model = converter.convert()

# Save the model.
tflite_model_path = f'{model_path}.tflite'
print("Saving", tflite_model_path)

with open(tflite_model_path, 'wb') as f:
  f.write(tflite_model)
