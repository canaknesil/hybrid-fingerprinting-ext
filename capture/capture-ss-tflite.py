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


import cwhardware
from tflite_type import *
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys


#PLATFORM = "CWLITEXMEGA"
PLATFORM = "CW308_STM32F4"

#fw_path = '../firmware/simpleserial-tflite/simpleserial-tflite-{}.hex'.format(PLATFORM)
fw_path = sys.argv[1]
input_data_path = sys.argv[2]
capture_path_prefix = sys.argv[3]

num_traces = 2000
avg_factor = 1

print("PLATFORM:", PLATFORM)
print("fw_path:", fw_path)
print("input_data_path:", input_data_path)
print("num_traces:", num_traces)
print("avg_factor:", avg_factor)

test_data = np.load(input_data_path)
print("input data shape:", test_data.shape)
print("input data type:", test_data.dtype)
assert num_traces <= test_data.shape[0]
test_data = test_data[:num_traces]


# Scope settings
n_samples = 24400 # For CW Lite, default=5000, max=24400
adc_clk_src = 'clkgen_x1'
decimation = 1 # ADC downsampling factor, sampling rate is 1/decimation of the sampling clock

# Invocation of hello_world_float.tflite takes 40.000 clock cycles.
#               mnist_model.tflite             80.000


#
# SETUP
#

hw = cwhardware.CWHardware()
hw.connect(PLATFORM)

# Confiture scope
hw.scope.default_setup();
hw.scope.adc.samples = n_samples
hw.scope.adc.decimate = decimation
hw.scope.clock.adc_src = adc_clk_src
time.sleep(0.1)
print("Target clock freq:", hw.scope.clock.clkgen_freq)
print("Sampling rate:", hw.scope.clock.adc_rate)

hw.program_target(fw_path)


#
# UTILITY
#

def multiply_list(lst):
    prod = 1
    for n in lst:
        prod *= n
    return prod


#
# GET MODEL INFO
#

model_info = hw.ss_read('f', 64)
if model_info is None:
    raise Exception("Reading model info failed!")

#print("readback model_info:", model_info)


def next_info():
    global offset
    info = int.from_bytes(model_info[offset:offset+4])
    offset += 4
    return info

offset = 0

input_n_dims = next_info()
input_shape = []
for i in range(input_n_dims):
    dim = next_info()
    input_shape.append(dim)

output_n_dims = next_info()
output_shape = []
for i in range(output_n_dims):
    dim = next_info()
    output_shape.append(dim)

print("input_shape:", input_shape)
print("output_shape:", output_shape)

input_type = next_info()
output_type = next_info()

print("input_type:", input_type, tflite_types[input_type])
print("output_type:", output_type, tflite_types[output_type])

input_type_np = tflite_type_to_np(input_type)
output_type_np = tflite_type_to_np(output_type)

correct_input_len = multiply_list(input_shape) * size_of_tflite_type(input_type)
correct_output_len = multiply_list(output_shape) * size_of_tflite_type(output_type)


#
# CAPTURE
#


def send_input_data(data):
    assert len(data) == correct_input_len

    # No need to send length as the input size is known.
    #data_len = len(data).to_bytes(4, "big")

    # Pad model with zeros until its length is multiple of 64.
    data += bytearray([0] * (-len(data) % 64))
    chunks = [data[i:i+64] for i in range(0, len(data), 64)]

    ret = hw.ss_write('g')
    if ret != 0:
        raise Exception("Input data transfer initialization unsuccessful!")
    
    for chunk in chunks:
        ret = hw.ss_write('h', chunk)
        if ret == 1:
            raise Exception("Input data pointer is null!")
        elif ret == 2:
            raise Exception("Input data area overflew!")
        if ret != 0:
            raise Exception("Error when sending input data!")


def receive_output_data():
    ret = hw.ss_write('j')
    if ret != 0:
        raise Exception("Output data pointer is null!")

    output_data = bytearray()

    n_chunk = correct_output_len // 64
    if correct_output_len % 64 != 0:
        n_chunk += 1

    for i in range(n_chunk):
        chunk = hw.ss_read('k', 64)
        output_data += chunk

    return output_data[:correct_output_len]


def infer(input_data):
    output_data, _ = infer_and_capture_trace(input_data, capture_trace=False)
    return output_data


def infer_and_capture_trace(input_data, capture_trace=True):
    input_data = bytearray(input_data)
    send_input_data(input_data)

    if capture_trace:
        hw.arm()

    # Invoke
    ret = hw.ss_write('i')
    if ret != 0:
        raise Exception("Invocation unsuccessful!")

    wave = None
    if capture_trace:
        wave = hw.capture()
        if wave is None:
            raise Exception("Capture unsuccessful!")
        
    output_data = receive_output_data()
    output_data = np.frombuffer(output_data, dtype=output_type_np).reshape(output_shape)
    return output_data, wave


# print("Test inference")
# input_data = np.full(input_shape, 0.5, dtype=input_type_np)
# output_data = infer(input_data)
# print(output_data)


# Check input data type and shape
assert input_shape[0] == 1
assert output_shape[0] == 1
assert list(test_data.shape[1:]) == list(input_shape[1:])
assert test_data.dtype == input_type_np


n_warmup_traces = 10
print(f"Capturing {n_warmup_traces} warming-up traces.")

warmup_traces = np.zeros([n_warmup_traces, n_samples], dtype=np.float32)
for i in range(n_warmup_traces):
    input_data = np.full(input_shape, 0.5, dtype=input_type_np)
    #input_data = np.random.rand(*input_shape).astype(input_type_np)

    output_data, trace = infer_and_capture_trace(input_data)
    warmup_traces[i] = trace.astype(np.float32)

#plt.figure()
#plt.plot(np.average(warmup_traces, axis=0))
#plt.show()
#sys.exit()


inputs = np.zeros([num_traces] + input_shape, dtype=input_type_np)
outputs = np.zeros([num_traces] + output_shape, dtype=output_type_np)
traces = np.zeros([num_traces, n_samples], dtype=np.float32)

traces_to_avg = np.zeros([avg_factor, n_samples], dtype=np.float32)
outputs_before_avg = np.zeros([avg_factor] + output_shape, dtype=output_type_np)

print("Capturing traces.")
for i in tqdm(range(num_traces)):
    #input_data = np.full(input_shape, 0.5, dtype=input_type_np)
    #input_data = np.random.rand(*input_shape).astype(input_type_np)
    input_data = test_data[i]
    inputs[i] = input_data

    if avg_factor == 1:
        output_data, trace = infer_and_capture_trace(input_data)
        traces[i] = trace.astype(np.float32)
        outputs[i] = output_data
    else:
        for j in range(avg_factor):
            output_data, trace = infer_and_capture_trace(input_data)
            traces_to_avg[j] = trace.astype(np.float32)
            outputs_before_avg[j] = output_data

        if not all(e == outputs_before_avg[0] for e in outputs_before_avg[1:]):
            print("Warning: Outputs before averaging are not identical!")
    
        traces[i] = np.average(traces_to_avg, axis=0)
        outputs[i] = outputs_before_avg[0]


np.save(capture_path_prefix + "_inputs.npy", inputs)
np.save(capture_path_prefix + "_outputs.npy", outputs)
np.save(capture_path_prefix + "_traces.npy", traces)

plt.figure()
#plt.plot(np.average(traces, axis=0))
plt.plot(traces[0])


#
# DISCONNECT
#

hw.disconnect()


#plt.show()
