import chipwhisperer as cw
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


PLATFORM = "CWLITEXMEGA"
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
