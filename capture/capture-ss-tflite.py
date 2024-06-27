import chipwhisperer as cw
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys


#PLATFORM = "CWLITEXMEGA"
PLATFORM = "CW308_STM32F4"
fw_path = '../firmware/simpleserial-tflite/simpleserial-tflite-{}.hex'.format(PLATFORM)

print("PLATFORM:", PLATFORM)
print("fw_path:", fw_path)

model_file = sys.argv[1]

print("model_file:", model_file)


#
# SETUP
#

# from Setup_Generic.ipynb
scope = cw.scope()
    
try:
    target = cw.target(scope)
except IOError:
    print("INFO: Caught exception on reconnecting to target - attempting to reconnect to scope first.")
    print("INFO: This is a work-around when USB has died without Python knowing. Ignore errors above this line.")
    scope = cw.scope()
    target = cw.target(scope)

print("INFO: Found ChipWhisperer😍")


if "STM" in PLATFORM or PLATFORM == "CWLITEARM" or PLATFORM == "CWNANO":
    prog = cw.programmers.STM32FProgrammer
elif PLATFORM == "CW303" or PLATFORM == "CWLITEXMEGA":
    prog = cw.programmers.XMEGAProgrammer
else:
    prog = None

time.sleep(0.05)
scope.default_setup()


def reset_target(scope):
    if PLATFORM == "CW303" or PLATFORM == "CWLITEXMEGA":
        scope.io.pdic = 'low'
        time.sleep(0.05)
        scope.io.pdic = 'high_z' #XMEGA doesn't like pdic driven high
        time.sleep(0.05)
    else:  
        scope.io.nrst = 'low'
        time.sleep(0.05)
        scope.io.nrst = 'high'
        time.sleep(0.05)


# from PA_CPA_1-Using_CW-Analyzer_for_CPA_Attack.ipynb
cw.program_target(scope, prog, fw_path)
time.sleep(1)
project = cw.create_project("projects/ss-tflite", overwrite = True)


#
# UTILITY
#

def disconnect():
    scope.dis()
    target.dis()


def ss_write(c, payload=[], timeout=5000):
    #print("Sending command '{}'".format(c), end="")
    target.simpleserial_write(c, payload)
    
    ret = target.simpleserial_wait_ack(timeout)
    if ret is None:
        raise Exception("Target failed to acknowledge!")

    return ret

    
def ss_read(c, payload_len, timeout=5000):
    #print("Sending command '{}'".format(c), end="")
    target.simpleserial_write(c, [])

    payload = target.simpleserial_read('r', payload_len)
    # target.simpleserial_read internally receives and checks ack

    #print(" -> payload")

    return payload


def multiply_list(lst):
    prod = 1
    for n in lst:
        prod *= n
    return prod


#
# SEND MODEL
#

with open(model_file, 'rb') as f:
    model = bytearray(f.read())

#model = bytearray([1, 2] * 32 + [3, 4] * 32 + [5])
    
print("model length:", len(model))
model_len = len(model).to_bytes(4, "big")

# Pad model with zeros until its length is multiple of 64.
model += bytearray([0] * (-len(model) % 64))
chunks = [model[i:i+64] for i in range(0, len(model), 64)]

print("Sending the model.")
ret = ss_write('a', model_len)
if ret != 0:
    raise Exception("Model memory allocation unsuccessful!")

for chunk in tqdm(chunks):
    ret = ss_write('b', chunk)
    if ret == 1:
        raise Exception("Model pointer is null!")
    elif ret == 2:
        raise Exception("Model area overflew!")
    if ret != 0:
        raise Exception("Error when sending model!")

        
#
# VERIFY MODEL
#

print("Reading the model back for verification.")
ret = ss_write('c')
if ret != 0:
    raise Exception("Model pointer is null!")

chunks2 = []
for chunk in tqdm(chunks):
    chunk2 = ss_read('d', 64)
    if chunk != chunk2:
        raise Exception("Readback model does not match the original!")
    
print("Verification successful.")


#
# INIT MODEL
#

print("Initializing the model.")
ret = ss_write('e')
if ret != 0:
    print("return code:", ret)
    raise Exception("Model initialization failed!")


model_info = ss_read('f', 64)
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

tflite_types = {
    0: "kTfLiteNoType",
    1: "kTfLiteFloat32",
    2: "kTfLiteInt32",
    3: "kTfLiteUInt8",
    4: "kTfLiteInt64",
    5: "kTfLiteString",
    6: "kTfLiteBool",
    7: "kTfLiteInt16",
    8: "kTfLiteComplex64",
    9: "kTfLiteInt8",
    10: "kTfLiteFloat16",
    11: "kTfLiteFloat64",
    12: "kTfLiteComplex128",
    13: "kTfLiteUInt64",
    14: "kTfLiteResource",
    15: "kTfLiteVariant",
    16: "kTfLiteUInt32",
    17: "kTfLiteUInt16",
    18: "kTfLiteInt4",
    19: "kTfLiteBFloat16",
}

tflite_type_sizes = {
    #"kTfLiteNoType": 0,
    "kTfLiteFloat32": 4,
    "kTfLiteInt32": 4,
    "kTfLiteUInt8": 1,
    "kTfLiteInt64": 8,
    #"kTfLiteString": 0,
    #"kTfLiteBool": 0,
    "kTfLiteInt16": 2,
    #"kTfLiteComplex64": 0,
    "kTfLiteInt8": 1,
    "kTfLiteFloat16": 2,
    "kTfLiteFloat64": 8,
    #"kTfLiteComplex128": 0,
    "kTfLiteUInt64": 8,
    #"kTfLiteResource": 0,
    #"kTfLiteVariant": 0,
    "kTfLiteUInt32": 4,
    "kTfLiteUInt16": 2,
    #"kTfLiteInt4": 0,
    #"kTfLiteBFloat16": 0,
}

def size_of_type(t):
    if type(t) == int:
        t = tflite_types[t]
    assert type(t) == str
    return tflite_type_sizes[t]
    

print("input_type:", input_type, tflite_types[input_type])
print("output_type:", output_type, tflite_types[output_type])

correct_input_len = multiply_list(input_shape) * size_of_type(input_type)
correct_output_len = multiply_list(output_shape) * size_of_type(output_type)


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

    ret = ss_write('g')
    if ret != 0:
        raise Exception("Input data transfer initialization unsuccessful!")
    
    for chunk in chunks:
        ret = ss_write('h', chunk)
        if ret == 1:
            raise Exception("Input data pointer is null!")
        elif ret == 2:
            raise Exception("Input data area overflew!")
        if ret != 0:
            raise Exception("Error when sending input data!")



input_data = bytearray([0] * 4)
print("Sending input data.")
send_input_data(input_data)



# print("Invoking.")
# ret = ss_write('i')
# if ret != 0:
#     raise Exception("Invocation unsuccessful!")



# ktp = cw.ktp.Basic()

# num_traces = 2

# print("Capturing traces...")
# for i in tqdm(range(num_traces)):
#     key, text = ktp.next()  # manual creation of a key, text pair can be substituted here
#     trace = cw.capture_trace(scope, target, text, key)
#     if trace is None:
#         continue
#     project.traces.append(trace)

# project.save()
# plt.plot(project.waves[0])

#
# DISCONNECT
#

disconnect()


#plt.show()
