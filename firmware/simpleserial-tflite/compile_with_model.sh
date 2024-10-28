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


# This script takes a .tflite model and produces a .hex file in the
# same directory as the model.

SS_TFLITE_PATH=`dirname $0`
echo SS_TFLITE_PATH=$SS_TFLITE_PATH

MODEL=$1
echo MODEL=$MODEL
echo

EXT="${MODEL##*.}"
if [ "$EXT" != "tflite" ]; then
    echo Model extension must be .tflite! Exiting.
    exit
fi

NEW_HEX="${MODEL%.tflite}.hex"
echo Creating $NEW_HEX...
echo

make -C $SS_TFLITE_PATH convert_model MODEL_TFLITE=$(realpath $MODEL)
make -C $SS_TFLITE_PATH
cp $SS_TFLITE_PATH/simpleserial-tflite-CW308_STM32F4.hex $NEW_HEX

echo
echo Created $NEW_HEX.
