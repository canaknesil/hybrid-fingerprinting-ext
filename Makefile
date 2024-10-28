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


# This makefile is not used for building. It prints information about
# the available scripts and their usage.


# Workspace W is where the input files to the available scripts should
# be and the output files will be produced.
W = workspace
$(info Workspace: $(W))


EMNIST_DIR = ../emnist/gzip
MODEL_NAME = mnist
X_SIZE = 7
Y_SIZE = 7
INDEP = 0
INIT = 0
MNIST_MODEL = $(MODEL_NAME)_$(X_SIZE)x$(Y_SIZE)

MODEL := $(MNIST_MODEL)
MODEL_MULTI := $(MODEL)_indep-$(INDEP)_init-$(INIT)
X_TEST := $(MODEL)_indep-$(INDEP)_x_test.npy
Y_TEST := $(MODEL)_indep-$(INDEP)_y_test.npy
RETRAIN_DATASET_PREFIX := $(MODEL)_indep-1
RETRAINED_MODEL := $(MODEL_MULTI)_retrained_ds-1
RAND_DATASET_PREFIX := $(MODEL)_rand

ORIGINAL := $(MODEL)_indep-0_init-0
SUSPECT := $(MODEL)_indep-0_init-0_snr-100
THIRD := $(MODEL)_indep-1_init-0

MODEL := $(MODEL_MULTI)
HEX_PREFIX := $(MODEL)
HEX := $(HEX_PREFIX).hex
TRACES_PREFIX := $(HEX_PREFIX)
INPUTS := $(TRACES_PREFIX)_inputs.npy
OUTPUTS := $(TRACES_PREFIX)_outputs.npy
TRACES := $(TRACES_PREFIX)_traces.npy


define print_target_info
	@echo
	@echo "Target: $@"
	@echo "Parameters:"
	@for s in $(foreach v,$(3),"$(v): $($(v))"); do echo "  $$s"; done
	@echo "Input files:"
	@for i in $(1); do echo "  $$i"; done
	@echo "Output files:"
	@for i in $(2); do echo "  $$i"; done
endef

define command
	@echo Command:
	@echo -n "  "
	@echo $(1)
endef



.PHONY: default all train_mnist modify_model retrain test_model convert_model_to_tflite compile_firmware gen_rand_mnist_data capture verify_inference compare_models detect_stolen

default:
	@echo "This makefile is not used for building. It prints information about the available scripts and their usage."


all: train_mnist train_mnist_multiple modify_model retrain test_model convert_model_to_tflite compile_firmware gen_rand_mnist_data capture verify_inference compare_models detect_stolen


train_mnist:
	$(call print_target_info,,$(W)/$(MNIST_MODEL) $(W)/$(MNIST_MODEL).keras,MODEL_NAME)
	$(call command,ipython tf/train-mnist.py $(W)/$(MODEL_NAME)) $(EMNIST_DIR)


train_mnist_multiple:
	$(call print_target_info,$(EMNIST_DIR)/emnist-digits-train-images-idx3-ubyte $(EMNIST_DIR)/emnist-digits-train-labels-idx1-ubyte $(EMNIST_DIR)/emnist-digits-test-images-idx3-ubyte $(EMNIST_DIR)/emnist-digits-test-labels-idx1-ubyte,$(shell bash -c "echo $(W)/$(MNIST_MODEL)_indep-{0..11}_{x,y}_{train,test}.npy") $(shell bash -c "echo $(W)/$(MNIST_MODEL)_indep-{0..11}_init-{0..1}{.keras,}"),MODEL_NAME EMNIST_DIR)
	$(call command,ipython tf/train-mnist-multiple.py $(W)/$(MODEL_NAME) $(EMNIST_DIR))


modify_model:
	$(call print_target_info,$(W)/$(MODEL).keras,$(shell bash -c "echo $(W)/$(MODEL)_snr-{1000,100,10}{.keras,}"),MODEL)
	$(call command,ipython tf/modify-model.py $(W)/$(MODEL))


retrain:
	$(call print_target_info,$(W)/$(MODEL).keras $(shell bash -c "echo $(W)/$(RETRAIN_DATASET_PREFIX)_{x,y}_{train,test}.npy"),$(W)/$(RETRAINED_MODEL) $(W)/$(RETRAINED_MODEL).keras $(W)/$(RETRAIN_DATASET_PREFIX)_y_train_from_victim.npy,MODEL RETRAIN_DATASET_PREFIX RETRAINED_MODEL)
	$(call command,ipython tf/retrain.py $(W)/$(MODEL) $(W)/$(RETRAIN_DATASET_PREFIX) $(W)/$(RETRAINED_MODEL)) 


test_model:
	$(call print_target_info,$(W)/$(MODEL).keras $(W)/$(X_TEST) $(W)/$(Y_TEST),,MODEL X_TEST Y_TEST)
	$(call command,ipython tf/test_model.py $(W)/$(MODEL) $(W)/$(X_TEST) $(W)/$(Y_TEST))


convert_model_to_tflite:
	$(call print_target_info,$(W)/$(MODEL) $(W)/$(MODEL).keras,$(W)/$(MODEL).tflite,MODEL)
	$(call command,ipython tf/convert-model-to-tflite.py $(W)/$(MODEL))


compile_firmware:
	$(call print_target_info,$(W)/$(MODEL).tflite,$(W)/$(HEX),MODEL)
	$(call command,bash firmware/simpleserial-tflite/compile_with_model.sh $(W)/$(MODEL).tflite)


gen_rand_mnist_data:
	$(call print_target_info,,$(W)/$(RAND_DATASET_PREFIX)_x_test.npy,RAND_DATASET_PREFIX)
	$(call command,ipython tf/gen-rand-mnist-test-data.py $(W)/$(RAND_DATASET_PREFIX))


capture:
	$(call print_target_info,$(W)/$(HEX) $(W)/$(X_TEST),$(W)/$(TRACES_PREFIX)_inputs.npy $(W)/$(TRACES_PREFIX)_outputs.npy $(W)/$(TRACES_PREFIX)_traces.npy,HEX X_TEST TRACES_PREFIX)
	$(call command,ipython capture/capture-ss-tflite.py $(W)/$(HEX) $(W)/$(X_TEST) $(W)/$(TRACES_PREFIX))


verify_inference:
	$(call print_target_info,$(W)/$(MODEL).tflite $(W)/$(INPUTS) $(W)/$(OUTPUTS),,MODEL INPUTS OUTPUTS)
	$(call command,ipython tf/infer-model-and-compare.py $(W)/$(MODEL).tflite $(W)/$(INPUTS) $(W)/$(OUTPUTS))


analysis_top:
	$(call command,ipython sca/analysis-top.py)


