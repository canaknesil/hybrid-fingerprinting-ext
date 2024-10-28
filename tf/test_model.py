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
import numpy as np


def test_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print("Number of test cases:", len(y_test))
    print("Test accuracy:", test_acc)


if __name__ == '__main__':
    model_path = sys.argv[1]
    print("model_path:", model_path)
    
    model_path_keras = model_path + ".keras"
    
    x_test_file = sys.argv[2]
    y_test_file = sys.argv[3]
    print("x_test_file:", x_test_file)
    print("y_test_file:", y_test_file)
    
    # Assuming .keras file exists.
    model = tf.keras.models.load_model(model_path_keras)
    model.summary()
    
    x_test = np.load(x_test_file)
    y_test = np.load(y_test_file)

    test_model(model, x_test, y_test)
    

