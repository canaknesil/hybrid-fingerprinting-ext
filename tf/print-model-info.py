# Print input and output types of a Tensorflow Lite model that was
# saved to a .tflite file.

# Code in this file was inspired from ChatGPT. Query: How can I learn
# input and output types of a Tensorflow Lite model saved in a .tflite
# file?

import tensorflow as tf
import sys


model_file = sys.argv[1]


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




