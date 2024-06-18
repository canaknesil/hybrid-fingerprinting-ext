extern "C" {
#include "hal.h"
#include "simpleserial.h"
#include <stdint.h>
#include <stdlib.h>
}
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


// uint8_t get_mask(uint8_t* m, uint8_t len)
// {
//    aes_indep_mask(m, len);
//    return 0x00;
// }

// uint8_t get_key(uint8_t* k, uint8_t len)
// {
//    aes_indep_key(k);
//    return 0x00;
// }

// uint8_t get_pt(uint8_t* pt, uint8_t len)
// {
//    aes_indep_enc_pretrigger(pt);

//    trigger_high();

// #ifdef ADD_JITTER
//    for (volatile uint8_t k = 0; k < (*pt & 0x0F); k++);
// #endif

//    aes_indep_enc(pt); /* encrypting the data block */
//    trigger_low();

//    aes_indep_enc_posttrigger(pt);

//    simpleserial_put('r', 16, pt);
//    return 0x00;
// }

// uint8_t reset(uint8_t* x, uint8_t len)
// {
//    // Reset key here if needed
//    return 0x00;
// }

// static uint16_t num_encryption_rounds = 10;

// uint8_t enc_multi_getpt(uint8_t* pt, uint8_t len)
// {
//    aes_indep_enc_pretrigger(pt);

//    for(unsigned int i = 0; i < num_encryption_rounds; i++){
//       trigger_high();
//       aes_indep_enc(pt);
//       trigger_low();
//    }

//    aes_indep_enc_posttrigger(pt);
//    simpleserial_put('r', 16, pt);
//    return 0;
// }

// uint8_t enc_multi_setnum(uint8_t* t, uint8_t len)
// {
//    //Assumes user entered a number like [0, 200] to mean "200"
//    //which is most sane looking for humans I think
//    num_encryption_rounds = t[1];
//    num_encryption_rounds |= t[0] << 8;
//    return 0;
// }


//
// WRITE/READ DATA IN 64-BIT CHUNKS
//

// Simpleserial supports receiving at most 64 bytes at a
// time. Receiving and sending data in 64 byte chunks.

// Hold data temporarily during write/read.  Keeping the pointer and
// deallocation is up to the caller, after the write is completed.

// For convenience, space allocated is a multiple of 64 bytes even
// though the provided length may be smaller. Remaining space is
// filled with zeros and can be read if necessary.

static uint8_t *data = 0;
static size_t data_len = 0;

static size_t offset = 0;


static uint8_t *write_reset(size_t len)
{
   if (len % 64 != 0)
      len = (len / 64 + 1) * 64;

   data = (uint8_t *) malloc(len * sizeof(uint8_t));

   if (data == 0)
      return 0;
   
   data_len = len;
   offset = 0;

   for (size_t i=0; i<data_len; i++)
      data[i] = 0;
   
   return data;
}


static uint8_t write_64(uint8_t *chunk)
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


static uint8_t read_reset(uint8_t *new_data, size_t new_len)
{
   if (new_data == 0)
      return 0x01;
   
   data = new_data;
   data_len = new_len;
   offset = 0;

   return 0x00;
}


static uint8_t *read_64()
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
// GET MODEL
//

static uint8_t *model;
static size_t model_len = 0;


static size_t convert_raw_to_uint(uint8_t* bytes, uint8_t bytes_len)
{
   size_t n = 0;

   // MSB first
   for (size_t i=0; i<bytes_len; i++) {
      n <<= 8;
      n += bytes[i];
   }

   return n;
}


static uint8_t get_model_reset(uint8_t* data, uint8_t len)
{
   if (model != 0)
      free(model);
   
   // data holds model length, MSB first.
   model_len = convert_raw_to_uint(data, len);
   model = write_reset(model_len);

   if (model == 0) {
      model_len = 0;
      return 0x01;
   } else {
      return 0x00;
   }
}


static uint8_t get_model_64(uint8_t* data, uint8_t len)
{
   return write_64(data);
}


static uint8_t put_model_reset(uint8_t* data, uint8_t len)
{
   return read_reset(model, model_len);
}


static uint8_t put_model_64(uint8_t* data, uint8_t len)
{
   if (model == 0)
      return 0x01;

   uint8_t *model_rb = read_64();

   if (model_rb == 0)
      return 0x02;

   simpleserial_put('r', 64, model_rb);
   
   return 0x00;
}


//
// TF LITE MICRO DRIVER
//

static uint8_t tflite_init_model(uint8_t* data, uint8_t len)
{
   const tflite::Model* tf_model = tflite::GetModel(model);

   if (tf_model == 0)
      return 0x01;

   // Following lines was inspired by ChatGPT. Query: How to use
   // tensorflow lite for microcontrollers in a C++ project?

   // Setup error reporter
   // static tflite::MicroErrorReporter micro_error_reporter;
   // tflite::ErrorReporter* error_reporter = &micro_error_reporter;

   // // Setup op resolver
   // static tflite::AllOpsResolver resolver;

   // // Setup tensor arena (this can be tailored to the model's needs)
   // constexpr int kTensorArenaSize = 10 * 1024;
   // static uint8_t tensor_arena[kTensorArenaSize];

   // // Setup interpreter
   // tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

   // // Allocate memory from tensor_arena for the model's tensors
   // TfLiteStatus allocate_status = interpreter.AllocateTensors();
   // if (allocate_status != kTfLiteOk) {
   //    error_reporter->Report("AllocateTensors() failed");
   //    return 1;
   // }

   // // Obtain pointers to the model's input and output tensors
   // TfLiteTensor* input = interpreter.input(0);
   // TfLiteTensor* output = interpreter.output(0);

   // // Fill input tensor with your data
   // input->data.f[0] = 1.0f;  // Example input

   // // Run inference
   // TfLiteStatus invoke_status = interpreter.Invoke();
   // if (invoke_status != kTfLiteOk) {
   //    error_reporter->Report("Invoke() failed");
   //    return 1;
   // }

   // // Process the output
   // float output_value = output->data.f[0];
   // error_reporter->Report("Output: %f", output_value);
   
   return 0x00;
}


//
// MAIN
//

int main(void)
{
   // uint8_t tmp[KEY_LENGTH] = {DEFAULT_KEY};

   platform_init();
   init_uart();
   trigger_setup();

   // aes_indep_init();
   // aes_indep_key(tmp);

   /* Uncomment this to get a HELLO message for debug */

   // putch('h');
   // putch('e');
   // putch('l');
   // putch('l');
   // putch('o');
   // putch('\n');

   simpleserial_init();

   // simpleserial_addcmd('k', 16, get_key);
   // simpleserial_addcmd('p', 16,  get_pt);
   // simpleserial_addcmd('x',  0,   reset);
   // simpleserial_addcmd_flags('m', 18, get_mask, CMD_FLAG_LEN);
   // simpleserial_addcmd('s', 2, enc_multi_setnum);
   // simpleserial_addcmd('f', 16, enc_multi_getpt);

   simpleserial_addcmd('a', 4, get_model_reset);
   simpleserial_addcmd('b', 64, get_model_64);
   simpleserial_addcmd('c', 0, put_model_reset);
   simpleserial_addcmd('d', 0, put_model_64);
   simpleserial_addcmd('e', 0, tflite_init_model);

   tflite::InitializeTarget();

   while(1)
      simpleserial_get();
}
