# Hey Emacs, this is a -*- makefile -*-
#----------------------------------------------------------------------------
#
# Makefile for ChipWhisperer SimpleSerial-AES Program
#
#----------------------------------------------------------------------------
# On command line:
#
# make all = Make software.
#
# make clean = Clean out built project files.
#
# make coff = Convert ELF to AVR COFF.
#
# make extcoff = Convert ELF to AVR Extended COFF.
#
# make program = Download the hex file to the device, using avrdude.
#                Please customize the avrdude settings below first!
#
# make debug = Start either simulavr or avarice as specified for debugging,
#              with avr-gdb or avr-insight as the front end for debugging.
#
# make filename.s = Just compile filename.c into the assembler code only.
#
# make filename.i = Create a preprocessed source file for use in submitting
#                   bug reports to the GCC project.
#
# To rebuild project do "make clean" then "make all".
#----------------------------------------------------------------------------


# To change the target platform, modify PLATFORM and CXX.

FIRMWAREPATH = ${HOME}/Programs/chipwhisperer/hardware/victims/firmware

#PLATFORM = CWLITEXMEGA
PLATFORM = CW308_STM32F4


# Target file name (without extension).
# This is the base name of the compiled .hex file.
TARGET = simpleserial-tflite

# List C/C++ source files here.
# Header files (.h) are automatically pulled in.

# Sources must NOT include .. as the directory structure is preserved
# in the object directory.

CPPSRC += simpleserial-tflite.cpp
#SRC += simpleserial-aes.c

MODEL_TFLITE = ../../models/hello_world_float.tflite
MODEL = models/model.cpp
CPPSRC += ${MODEL}


#CPPSRC += $(filter-out %_test.cc,$(notdir $(wildcard ${TFPATH}/tensorflow/lite/micro/*.cc)))

VPATH += :third/tflite-micro

CPPSRC += tensorflow/lite/micro/debug_log.cc
#CPPSRC += tensorflow/lite/micro/fake_micro_context.cc
CPPSRC += tensorflow/lite/micro/flatbuffer_utils.cc
CPPSRC += tensorflow/lite/micro/memory_helpers.cc
CPPSRC += tensorflow/lite/micro/micro_allocation_info.cc
CPPSRC += tensorflow/lite/micro/micro_allocator.cc
CPPSRC += tensorflow/lite/micro/micro_context.cc
CPPSRC += tensorflow/lite/micro/micro_interpreter.cc
CPPSRC += tensorflow/lite/micro/micro_interpreter_context.cc
CPPSRC += tensorflow/lite/micro/micro_interpreter_graph.cc
CPPSRC += tensorflow/lite/micro/micro_log.cc
CPPSRC += tensorflow/lite/micro/micro_op_resolver.cc
#CPPSRC += tensorflow/lite/micro/micro_profiler.cc
CPPSRC += tensorflow/lite/micro/micro_resource_variable.cc
#CPPSRC += tensorflow/lite/micro/micro_time.cc
CPPSRC += tensorflow/lite/micro/micro_utils.cc
#CPPSRC += tensorflow/lite/micro/mock_micro_graph.cc
#CPPSRC += tensorflow/lite/micro/recording_micro_allocator.cc
CPPSRC += tensorflow/lite/micro/system_setup.cc
#CPPSRC += tensorflow/lite/micro/test_helper_custom_ops.cc
#CPPSRC += tensorflow/lite/micro/test_helpers.cc

CPPSRC += tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc

CPPSRC += tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc
CPPSRC += tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc

CPPSRC += tensorflow/lite/micro/memory_planner/linear_memory_planner.cc
CPPSRC += tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc

CPPSRC += tensorflow/lite/micro/kernels/fully_connected.cc
CPPSRC += tensorflow/lite/micro/kernels/fully_connected_common.cc
CPPSRC += tensorflow/lite/micro/kernels/kernel_util.cc
CPPSRC += tensorflow/lite/micro/kernels/conv.cc
CPPSRC += tensorflow/lite/micro/kernels/conv_common.cc
CPPSRC += tensorflow/lite/micro/kernels/depthwise_conv.cc
CPPSRC += tensorflow/lite/micro/kernels/depthwise_conv_common.cc
CPPSRC += tensorflow/lite/micro/kernels/reshape.cc
CPPSRC += tensorflow/lite/micro/kernels/reshape_common.cc
CPPSRC += tensorflow/lite/micro/kernels/softmax.cc
CPPSRC += tensorflow/lite/micro/kernels/softmax_common.cc
CPPSRC += tensorflow/lite/micro/kernels/pooling.cc
CPPSRC += tensorflow/lite/micro/kernels/pooling_common.cc

CPPSRC += tensorflow/lite/core/c/common.cc

CPPSRC += tensorflow/lite/core/api/flatbuffer_conversions.cc
CPPSRC += tensorflow/lite/core/api/error_reporter.cc

CPPSRC += tensorflow/lite/kernels/kernel_util.cc
CPPSRC += tensorflow/lite/kernels/internal/common.cc
CPPSRC += tensorflow/lite/kernels/internal/portable_tensor_utils.cc
CPPSRC += tensorflow/lite/kernels/internal/quantization_util.cc

CPPSRC += tensorflow/lite/schema/schema_utils.cc


EXTRAINCDIRS += third/tflite-micro
EXTRAINCDIRS += third/flatbuffers/include
EXTRAINCDIRS += third/gemmlowp
EXTRAINCDIRS += third/ruy


#CXX = avr-g++
CXX = arm-none-eabi-g++

EXTRA_OPTS = NO_EXTRA_OPTS
CFLAGS += -D$(EXTRA_OPTS)


# Add simpleserial project to build
SS_VER = SS_VER_1_1

include Makefile.firmware

