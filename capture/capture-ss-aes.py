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


import chipwhisperer as cw
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


#PLATFORM = "CWLITEXMEGA"
PLATFORM = "CW308_STM32F4"
fw_path = '../firmware/simpleserial-aes/simpleserial-aes-{}.hex'.format(PLATFORM)

print("PLATFORM: ", PLATFORM)
print("fw_path: ", fw_path)


#
# SETUP
#

# from Setup_Generic.ipynb
try:
    if not scope.connectStatus:
        scope.con()
except NameError:
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
#scope.adc.samples = 24400 # 24400 is max for CWLITE
#scope.adc.decimate = 4
#scope.clock.adc_src = "clkgen_x1"


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
project = cw.create_project("projects/Tutorial_B5", overwrite = True)


#
# CAPTURE
#

ktp = cw.ktp.Basic()

num_traces = 2

print("Capturing traces...")
for i in tqdm(range(num_traces)):
    key, text = ktp.next()  # manual creation of a key, text pair can be substituted here
    trace = cw.capture_trace(scope, target, text, key)
    if trace is None:
        continue
    project.traces.append(trace)

project.save()
plt.plot(project.waves[0])

#
# DISCONNECT
#

scope.dis()
target.dis()


plt.show()
