import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import tensorflow as tf


model_path = sys.argv[1]

# model_path shouldn't end with / because .tflite extension will be
# added to the end.
while model_path[-1] == '/':
    model_path = model_path[:-1]
    
print("model_path:", model_path)

model_path_keras = model_path + ".keras"


# Evaluate model

# Assuming .keras file exists.
model = tf.keras.models.load_model(model_path_keras)
model.summary()
# TODO: Decide whether the model operations are support on TfLite Micro.


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
tflite_model = converter.convert()

# Save the model.
tflite_model_path = f'{model_path}.tflite'
print("Saving", tflite_model_path)

with open(tflite_model_path, 'wb') as f:
  f.write(tflite_model)
