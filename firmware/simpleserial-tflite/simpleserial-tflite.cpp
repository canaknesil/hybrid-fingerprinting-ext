extern "C" {
#include "hal.h"
#include "simpleserial.h"
#include <stdint.h>
#include <stdlib.h>
}


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


// Simpleserial supports receiving at most 64 bytes at a
// time. Receiving the models in 64 byte chunks.

// For now using the main memory for the model for testing.
static uint8_t model[128];

static size_t model_write_addr_offset = 0;
static size_t model_read_addr_offset = 0;

static uint8_t get_model_reset(uint8_t* data, uint8_t len)
{
   model_write_addr_offset = 0;
   return 0x00;
}

static uint8_t get_model_64(uint8_t* data, uint8_t len)
{
   for (size_t i=0; i<64; i++)
      model[model_write_addr_offset++] = data[i];
   
   return 0x00;
}

static uint8_t put_model_reset(uint8_t* data, uint8_t len)
{
   model_read_addr_offset = 0;
   return 0x00;
}

static uint8_t put_model_64(uint8_t* data, uint8_t len)
{
   simpleserial_put('r', 64, model + model_read_addr_offset);
   model_read_addr_offset += 64;
   
   return 0x00;
}


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

   simpleserial_addcmd('a', 0, get_model_reset);
   simpleserial_addcmd('b', 64, get_model_64);
   simpleserial_addcmd('c', 0, put_model_reset);
   simpleserial_addcmd('d', 0, put_model_64);

   while(1)
      simpleserial_get();
}
