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
