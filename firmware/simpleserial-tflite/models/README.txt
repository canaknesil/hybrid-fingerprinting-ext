This directory should contain .tflite models in C/C++ array format,
converted via, e.g., xxd and hand-modified. Desired model should be
added to the sources list in Makefile.

Converted model files are not added to version control as .tflite
versions are already in.

File format:

#include "model.h"

namespace ss_tflite {

const unsigned char model[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, ...
};

const unsigned int model_len = 3164;


