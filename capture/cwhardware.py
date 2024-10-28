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


class CWHardware:
    # from Setup_Generic.ipynb
    def connect(self, platform):
        PLATFORM = platform
        
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

        self.scope = scope
        self.target = target
        self.target_programmer = prog
        self.platform = PLATFORM
                
        time.sleep(0.05)
        return True
    
    
    def reset_target(self):
        PLATFORM = self.platform
        scope = self.scope
        
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

        return True
        
    
    def program_target(self, fw_path):
        scope = self.scope
        prog = self.target_programmer
        
        # from PA_CPA_1-Using_CW-Analyzer_for_CPA_Attack.ipynb
        cw.program_target(scope, prog, fw_path)
        time.sleep(1)
        return True

    
    def disconnect(self):
        self.scope.dis()
        self.target.dis()

        
    def ss_write(self, c, payload=[], timeout=5000):
        #print("Sending command '{}'".format(c), end="")
        self.target.simpleserial_write(c, payload)
        
        ret = self.target.simpleserial_wait_ack(timeout)
        if ret is None:
            raise Exception("Target failed to acknowledge!")
    
        return ret
    
        
    def ss_read(self, c, payload_len, timeout=5000):
        #print("Sending command '{}'".format(c), end="")
        self.target.simpleserial_write(c, [])
    
        payload = self.target.simpleserial_read('r', payload_len)
        # target.simpleserial_read internally receives and checks ack
    
        #print(" -> payload")
    
        return payload

    def arm(self):
        self.scope.arm()
        
    def capture(self):
        scope = self.scope
        target = self.target
        
        ret = scope.capture(poll_done=False)
        
        i = 0
        while not target.is_done():
            i += 1
            time.sleep(0.05)
            if i > 100:
                print("Warning: Target did not finish operation!")
                return None
        
        if ret:
            print("Warning: Timeout happened during capture!")
            return None
        
        wave = scope.get_last_trace(as_int=False)
        
        if len(wave) >= 1:
            return wave
        else:
            return None
