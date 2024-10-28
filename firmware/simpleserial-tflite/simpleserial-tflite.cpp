// MIT License

// Copyright (c) 2024 Can Aknesil

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


extern "C" {
#include "hal.h"
#include "simpleserial.h"
#include <stdint.h>
#include <stdlib.h>
}
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
//#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/core/c/common.h"
//#include "tensorflow/lite/micro/micro_log.h"
//#include "tensorflow/lite/micro/micro_profiler.h"
//#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "model.h"


namespace ss_tflite {

   //
   // UTILS
   //

   size_t convert_raw_to_uint(uint8_t* bytes, uint8_t bytes_len)
   {
      size_t n = 0;

      // MSB is bytes[0]
      for (size_t i=0; i<bytes_len; i++) {
	 n <<= 8;
	 n += bytes[i];
      }

      return n;
   }


   void convert_uint32_to_raw(uint8_t *bytes, uint32_t n)
   {
      // MSB is bytes[0]
      for (int i=3; i>=0; i--) {
	 bytes[i] = n % 256;
	 n >>= 8;
      }
   }

   
   void ss_put_uint32(uint32_t n)
   {
      uint8_t size_raw[4];
      convert_uint32_to_raw(size_raw, n);
      simpleserial_put('r', 4, size_raw);
   }


   size_t sizeof_TfLiteType(TfLiteType type) {
      switch (type) {
      case kTfLiteFloat32:
	 return sizeof(float);
      case kTfLiteInt32:
	 return sizeof(int32_t);
      case kTfLiteUInt8:
	 return sizeof(uint8_t);
      case kTfLiteInt64:
	 return sizeof(int64_t);
      case kTfLiteBool:
	 return sizeof(bool);
      case kTfLiteInt16:
	 return sizeof(int16_t);
	 // case kTfLiteComplex64:
	 // 	 return sizeof(std::complex<float>);
      case kTfLiteInt8:
	 return sizeof(int8_t);
      case kTfLiteFloat16:
	 return sizeof(uint16_t);  // float16 is usually stored as a uint16_t
      case kTfLiteFloat64:
	 return sizeof(double);
	 // case kTfLiteComplex128:
	 // 	 return sizeof(std::complex<double>);
      case kTfLiteUInt64:
	 return sizeof(uint64_t);
      case kTfLiteUInt32:
	 return sizeof(uint32_t);
      case kTfLiteUInt16:
	 return sizeof(uint16_t);
      default:
	 return 0;
      }
   }


   size_t multiple_of_64(size_t len)
   {
      if (len % 64 != 0)
	 len = (len / 64 + 1) * 64;
      return len;
   }


   //
   // WRITE/READ DATA IN 64-BIT CHUNKS
   //

   // Simpleserial supports receiving at most 64 bytes at a
   // time. Receiving and sending data in 64 byte chunks.

   // Hold pointer to data temporarily during write/read.  Keeping the
   // pointer and deallocation is up to the caller, after the write is
   // completed.

   // For convenience, space allocated is a multiple of 64 bytes even
   // though the provided length may be smaller. Remaining space is
   // filled with zeros and can be read if necessary.

   uint8_t *data = 0;
   size_t data_len = 0;

   size_t offset = 0;


   // len must be a multiple of 64.
   void write_reset_preallocated(uint8_t *ptr, size_t len)
   {
      if (ptr == 0) {
	 data = 0;
	 data_len = 0;
	 return;
      }

      if (len % 64 != 0) {
	 data = 0;
	 data_len = 0;
	 return;
      }

      data = ptr;
      data_len = len;
      offset = 0;

      for (size_t i=0; i<data_len; i++)
	 data[i] = 0;
   }
   

   uint8_t *write_reset(size_t len)
   {
      len = multiple_of_64(len);
      uint8_t *ptr = (uint8_t *) malloc(len * sizeof(uint8_t));
      
      write_reset_preallocated(ptr, len);
   
      return ptr;
   }


   uint8_t write_64(uint8_t *chunk)
   {
      if (data == 0)
	 return 0x01;

      if (offset >= data_len)
	 return 0x02;

      for (size_t i=0; i<64; i++) {
	 if (offset >= data_len)
	    break;
	 data[offset++] = chunk[i];
      }
   
      return 0x00;
   }


   uint8_t read_reset(uint8_t *new_data, size_t new_len)
   {
      if (new_data == 0)
	 return 0x01;
   
      data = new_data;
      data_len = new_len;
      offset = 0;

      return 0x00;
   }


   uint8_t *read_64()
   {
      if (data == 0)
	 return 0;

      if (offset >= data_len)
	 return 0;

      uint8_t *data_rb = data + offset;
      offset += 64;
   
      return data_rb;
   }


   //
   // GET INPUT DATA
   //

   // Input data buffer is allocated and deallocated during model
   // initialization.
   uint8_t *input_data = 0;
   size_t input_data_len = 0;


   uint8_t get_input_data_reset(uint8_t* data, uint8_t len)
   {
      if (input_data == 0)
	 return 0x01;
      
      write_reset_preallocated(input_data, multiple_of_64(input_data_len));

      return 0x00;
   }


   uint8_t get_input_data_64(uint8_t* data, uint8_t len)
   {
      return write_64(data);
   }


   //
   // PUT OUTPUT DATA
   //

   // Output data buffer is allocated and deallocated during model IO
   // preparation.
   uint8_t *output_data;
   size_t output_data_len = 0;


   uint8_t put_output_data_reset(uint8_t* data, uint8_t len)
   {
      return read_reset(output_data, output_data_len);
   }


   uint8_t put_output_data_64(uint8_t* data, uint8_t len)
   {
      if (output_data == 0)
	 return 0x01;

      uint8_t *output_data_rb = read_64();

      if (output_data_rb == 0)
	 return 0x02;

      simpleserial_put('r', 64, output_data_rb);
   
      return 0x00;
   }


   //
   // TF LITE MICRO DRIVER
   //

   tflite::MicroInterpreter *interpreter;
   
   size_t model_input_n_dims = 0;
   size_t model_input_dims[16] = {0};
   TfLiteType model_input_type;
   
   size_t model_output_n_dims = 0;
   size_t model_output_dims[16] = {0};
   TfLiteType model_output_type;

   
   uint8_t prepare_model_io(tflite::MicroInterpreter &interpreter)
   {
      // Save model input/output info.
      TfLiteTensor* input = interpreter.input(0);
      TfLiteTensor* output = interpreter.output(0);

      model_input_n_dims = input->dims->size; // actual type is int
      model_output_n_dims = output->dims->size;

      for (size_t i=0; i<model_input_n_dims; i++)
	 model_input_dims[i] = input->dims->data[i];

      for (size_t i=0; i<model_output_n_dims; i++)
	 model_output_dims[i] = output->dims->data[i];

      model_input_type = input->type;
      model_output_type = output->type;

      // Allocate input/output data buffers
      if (input_data != 0)
	 free(input_data);
      if (output_data != 0)
	 free(output_data);
      
      input_data_len = 1;
      for (size_t i=0; i<model_input_n_dims; i++)
	 input_data_len *= model_input_dims[i];
      input_data_len *= sizeof_TfLiteType(model_input_type);

      output_data_len = 1;
      for (size_t i=0; i<model_output_n_dims; i++)
	 output_data_len *= model_output_dims[i];
      output_data_len *= sizeof_TfLiteType(model_output_type);

      input_data = (uint8_t *) malloc(multiple_of_64(input_data_len) * sizeof(uint8_t));
      output_data = (uint8_t *) malloc(multiple_of_64(output_data_len) * sizeof(uint8_t));

      return 0x00;
   }


   uint8_t tflite_put_model_info(uint8_t* data, uint8_t len)
   {
      if (interpreter->initialization_status() != kTfLiteOk)
	 return 0x01;
      
      uint8_t info[64];
      size_t offset = 0;
      
      convert_uint32_to_raw(&info[offset], model_input_n_dims);
      offset += 4;
      
      for (size_t i=0; i<model_input_n_dims; i++) {
	 uint32_t dim = model_input_dims[i];
	 convert_uint32_to_raw(&info[offset], dim);
	 offset += 4;
      }

      convert_uint32_to_raw(&info[offset], model_output_n_dims);
      offset += 4;
      
      for (size_t i=0; i<model_output_n_dims; i++) {
	 uint32_t dim = model_output_dims[i];
	 convert_uint32_to_raw(&info[offset], dim);
	 offset += 4;
      }

      convert_uint32_to_raw(&info[offset], (uint32_t) model_input_type);
      offset += 4;

      convert_uint32_to_raw(&info[offset], (uint32_t) model_output_type);
      offset += 4;

      simpleserial_put('r', 64, info);

      return 0x00;
   }


   uint8_t tflite_invoke(uint8_t* data, uint8_t len)
   {
      TfLiteTensor* input = interpreter->input(0);
      TfLiteTensor* output = interpreter->output(0);

      // This works only for float32.
      for (size_t i=0; i<input_data_len/sizeof(float); i++) {
	 float *input_data_casted = (float *) input_data;
	 input->data.f[i] = input_data_casted[i];
      }
      //input->data.f[0] = 0.0;

      trigger_high();
      TfLiteStatus status = interpreter->Invoke();
      trigger_low();
      
      if (status != kTfLiteOk)
	 return 0x01;

      // This works only for float32.
      for (size_t i=0; i<output_data_len/sizeof(float); i++) {
	 float *output_data_casted = (float *) output_data;
	 output_data_casted[i] = output->data.f[i];
      }
      
      return 0x00;
   }
}



//
// MAIN
//

using namespace ss_tflite;

int main(void)
{
   // Initialize platform
   platform_init();
   init_uart();
   trigger_setup();

   simpleserial_init();
   simpleserial_addcmd('f', 0, tflite_put_model_info);
   simpleserial_addcmd('g', 0, get_input_data_reset);
   simpleserial_addcmd('h', 64, get_input_data_64);
   simpleserial_addcmd('i', 0, tflite_invoke);
   simpleserial_addcmd('j', 0, put_output_data_reset);
   simpleserial_addcmd('k', 0, put_output_data_64);

   // Initialize tflite
   tflite::InitializeTarget();

   tflite::MicroMutableOpResolver<6> resolver;
   resolver.AddFullyConnected();
   resolver.AddConv2D();
   resolver.AddDepthwiseConv2D();
   resolver.AddReshape();
   resolver.AddSoftmax();
   resolver.AddAveragePool2D();

   constexpr int tensor_arena_size = 50 * 1024;
   uint8_t tensor_arena[tensor_arena_size];

   const tflite::Model* tf_model = tflite::GetModel(model);

   tflite::MicroInterpreter interpreter_local(tf_model, resolver, tensor_arena, tensor_arena_size);
   interpreter = &interpreter_local;

   interpreter_local.AllocateTensors();
   prepare_model_io(interpreter_local);

   while(1)
      simpleserial_get();
}
