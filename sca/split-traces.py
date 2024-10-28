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


# This script is used to split a trace file into two. Currently not used.

from glob import glob
import os
import re
import sys
import numpy as np


workspace = "workspace"

# With help from ChatGPT
def natural_sort_key(s):
    # Split the string into a list of integers and non-integer parts
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def ls_re(pattern):
    items = os.listdir(workspace)
    items = [i for i in items if re.search(pattern, i)]
    return sorted(items, key=natural_sort_key)


def save(f, a):
    print(f)
    np.save(f, a)


models = ls_re("^mnist_7x7_indep-\d+_init-\d+$")
models_part1 = list(map(lambda s: s + "_part1", models))
models_part2 = list(map(lambda s: s + "_part2", models))


for a, b, c in zip(models, models_part1, models_part2):
    print(a)
    inputs = np.load(workspace + "/" + a + "_inputs.npy")
    outputs = np.load(workspace + "/" + a + "_outputs.npy")
    traces = np.load(workspace + "/" + a + "_traces.npy")

    perm = np.random.permutation(len(inputs))
    half = len(inputs) // 2
    idx1 = perm[:half]
    idx2 = perm[half:]

    inputs1 = inputs[idx1]
    inputs2 = inputs[idx2]
    outputs1 = outputs[idx1]
    outputs2 = outputs[idx2]
    traces1 = traces[idx1]
    traces2 = traces[idx2]

    save(workspace + "/" + b + "_inputs.npy", inputs1)
    save(workspace + "/" + b + "_outputs.npy", outputs1)
    save(workspace + "/" + b + "_traces.npy", traces1)

    save(workspace + "/" + c + "_inputs.npy", inputs2)
    save(workspace + "/" + c + "_outputs.npy", outputs2)
    save(workspace + "/" + c + "_traces.npy", traces2)
    
