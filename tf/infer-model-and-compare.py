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
import numpy as np
import sys


model_file = sys.argv[1]
inputs_file = sys.argv[2]
outputs_file = sys.argv[3]


interpreter = tf.lite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#tensor_details = interpreter.get_tensor_details()

input_shape = input_details[0]['shape']


def infer(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


# Test inference
# input_data = np.full(input_shape, 0.5, dtype=np.float32)
# #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# output_data = infer(input_data)
# print(output_data)

inputs = np.load(inputs_file)
outputs_micro = np.load(outputs_file)

outputs = np.zeros_like(outputs_micro)
for i, input_data in enumerate(inputs):
    outputs[i] = infer(input_data)

max_error = np.max(np.abs(outputs_micro - outputs))
print("max_error:", max_error)
print("For comparison, largest single floating point error is", np.float64(1.1920929e-7))



