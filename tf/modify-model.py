import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import tensorflow as tf
import numpy as np
import util


model_path = sys.argv[1]

# model_path shouldn't end with /.
while model_path[-1] == '/':
    model_path = model_path[:-1]
    
print("model_path:", model_path)

model_path_keras = model_path + ".keras"

# std of weights, divided by std of noise with zero mean, when adding
# noise to the weights. Specify integer values as it will be present
# in filename.
snr_list = [1000, 100, 10]



# Assuming .keras file exists.
model = tf.keras.models.load_model(model_path_keras)
model.summary()


def check_weights(w):
    assert type(w) == list
    for e in w:
        assert type(e) == np.ndarray


def copy_weights(w):
    w2 = list(map(np.copy, w))
    return w2


original_weights = model.get_weights() # list of numpy arrays
check_weights(original_weights)

weights = copy_weights(original_weights)

for e in weights:
    avg = np.average(e)
    std = np.std(e)
    print(f"avg={avg}, std={std}")

flat_weights = np.concatenate(list(map(lambda a: a.flatten(), weights)))
weights_avg = np.average(flat_weights)
weights_std = np.std(flat_weights)
print(f"Overall avg={weights_avg}, std={weights_std}")


for snr in snr_list:
    for e in weights:
        noise = np.random.normal(loc=0.0, scale=weights_std / snr, size=e.shape)
        e += noise

    model.set_weights(weights)
    new_model_file = f"{model_path}_snr-{snr}"
    #print(new_model_file)
    util.save_model(model, new_model_file)


