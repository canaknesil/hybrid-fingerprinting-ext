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
    

