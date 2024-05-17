import chipwhisperer as cw
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys


PLATFORM = "CWLITEXMEGA"
fw_path = '../firmware/simpleserial-tflite/simpleserial-tflite-{}.hex'.format(PLATFORM)

print("PLATFORM: ", PLATFORM)
print("fw_path: ", fw_path)


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


def ss_write(c, payload=[], timeout=2000):
    #print("Sending command '{}'".format(c), end="")
    target.simpleserial_write(c, payload)
    
    ret = target.simpleserial_wait_ack(timeout)
    if ret is None:
        raise Exception("Target failed to acknowledge!")

    return ret

    
def ss_read(c, payload_len, timeout=2000):
    #print("Sending command '{}'".format(c), end="")
    target.simpleserial_write(c, [])

    payload = target.simpleserial_read('r', 64)
    # target.simpleserial_read internally receives and checks ack

    #print(" -> payload")

    return payload


#
# SEND MODEL
#

if ss_write('g') != 0:
    raise Exception("Model memory allocation unsuccessful!")

model = bytearray([1, 2] * 32 + [3, 4] * 30)

# Pad model with zeros until its length is multiple of 64.
model += bytearray([0] * (-len(model) % 64))
chunks = [model[i:i+64] for i in range(0, len(model), 64)]

print("Sending the model.")
ss_write('a')
for chunk in tqdm(chunks):
    ss_write('b', chunk)


#
# VERIFY MODEL
#

print("Reading the model back for verification.")
chunks2 = []
ss_write('c')
for chunk in tqdm(chunks):
    chunk2 = ss_read('d', 64)
    if chunk != chunk2:
        raise Exception("Readback model does not match the original!")
    
print("Verification successful.")


#
# CAPTURE
#

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
