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


# Print input and output types of a Tensorflow Lite model that was
# saved to a .tflite file.

# Code in this file was inspired from ChatGPT. Query: How can I learn
# input and output types of a Tensorflow Lite model saved in a .tflite
# file?

import tensorflow as tf
import sys


model_file = sys.argv[1]


def print_model_info_tflite(model_file):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    
    # Get input and output tensor details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    
    # Print input details
    # print("Input Details:")
    # for input_tensor in input_details:
    #     print(f"Name: {input_tensor['name']}")
    #     print(f"Shape: {input_tensor['shape']}")
    #     print(f"Type: {input_tensor['dtype']}")
    #     print()
    
    # Print output details
    # print("Output Details:")
    # for output_tensor in output_details:
    #     print(f"Name: {output_tensor['name']}")
    #     print(f"Shape: {output_tensor['shape']}")
    #     print(f"Type: {output_tensor['dtype']}")
    #     print()
    
    print("Tensor Details:")
    for output_tensor in tensor_details:
        print(f"Name: {output_tensor['name']}")
        print(f"Shape: {output_tensor['shape']}")
        print(f"Type: {output_tensor['dtype']}")
        print()


def print_model_info(model_file):
    model = tf.keras.models.load_model(model_file)
    model.summary()


if model_file.endswith(".tflite"):
    print_model_info_tflite(model_file)
else:
    print_model_info(model_file)



