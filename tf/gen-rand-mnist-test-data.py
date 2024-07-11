import numpy as np
import sys
import util

dataset_prefix = util.remove_trailing_slash(sys.argv[1])
print("dataset_prefix:", dataset_prefix)

x_test_fname = dataset_prefix + "_x_test.npy"

x_test_shape = (10000, 7, 7)
x_test_dtype = np.float32

# range [0, 1]
x_test = np.random.rand(*x_test_shape).astype(x_test_dtype)

print("x_test.shape:", x_test.shape)
print("x_test.dtype:", x_test.dtype)

print("Saving to", x_test_fname)
np.save(x_test_fname, x_test)

