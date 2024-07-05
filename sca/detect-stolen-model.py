import sys
import numpy as np


original_prefix = sys.argv[1]
suspect_prefix = sys.argv[2]
third_prefix = sys.argv[3]
correct_outputs_file = sys.argv[4]

print("original_prefix:", original_prefix)
print("suspect_prefix:", suspect_prefix)
print("third_prefix:", third_prefix)
print("correct_outputs_file:", correct_outputs_file)

outputs_original = np.load(original_prefix + "_outputs.npy")
outputs_suspect = np.load(suspect_prefix + "_outputs.npy")
outputs_third = np.load(third_prefix + "_outputs.npy")
correct_outputs = np.load(correct_outputs_file)

print("outputs_original.shape:", outputs_original.shape)
print("outputs_suspect.shape:", outputs_suspect.shape)
print("outputs_third.shape:", outputs_third.shape)
print("correct_outputs.shape:", correct_outputs.shape)

print("Removing dimensions of size 1.")
outputs_original = np.squeeze(outputs_original)
outputs_suspect = np.squeeze(outputs_suspect)
outputs_third = np.squeeze(outputs_third)
correct_outputs = np.squeeze(correct_outputs)


assert outputs_original.shape[0] <= correct_outputs.shape[0]
if outputs_original.shape[0] < correct_outputs.shape[0]:
    print("Warning: Traces were collected from a subset of test data. Choping test data.")
    correct_outputs = correct_outputs[:outputs_original.shape[0],:]

outputs_list = [outputs_original, outputs_suspect, outputs_third, correct_outputs]
for x in outputs_list[1:]:
    assert x.shape == outputs_list[0].shape
    

def logits_to_predictions(logits):
    assert len(logits.shape) == 2
    predictions = np.zeros(logits.shape[0], dtype=int)
    for i in range(logits.shape[0]):
        predictions[i] = np.argmax(logits[i])
    return predictions


logits_list = outputs_list
outputs_list = list(map(logits_to_predictions, outputs_list))

    
# TODO: Consider traces


def isequal(a1, a2):
    res = np.zeros(a1.shape[0], dtype=bool)
    for i in range(a1.shape[0]):
        x = a1[i] == a2[i]
        if len(np.shape(x)) > 0:
            x = all(x)
        res[i] = x
    return res


def notequal(a1, a2):
    return np.logical_not(isequal(a1, a2))


def detect_direct_copy(outputs_original, outputs_suspect, outputs_third, correct_outputs):
    size = len(outputs_original)
    n_similar = np.count_nonzero(isequal(outputs_original, outputs_suspect))
    n_similar_third = np.count_nonzero(isequal(outputs_original, outputs_third))
    similarity_rate = n_similar / size
    similarity_rate_third = n_similar_third / size
    print(f"{n_similar} out of {size} ({similarity_rate*100}%) predictions by suspect are similar to the original model.")
    if n_similar < size:
        print("The suspect IS NOT a direct copy.")
    else:
        print("The suspect may be a direct copy.")
    print(f"{n_similar_third} out of {size} ({similarity_rate_third*100}%) predictions by the 3rd-party model are similar to the original model.")


def logit_distance(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))


def avg_logit_distance(a, b):
    dist = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        dist[i] = logit_distance(a[i], b[i])
    return np.average(dist)


def find_logits_similarity(original, suspect, third, correct):
    avg_dist_original_vs_suspect = avg_logit_distance(original, suspect)
    avg_dist_original_vs_third = avg_logit_distance(original, third)
    print("Average logit distance of original vs. suspect:", avg_dist_original_vs_suspect)
    print("Average logit distance of original vs. 3rd-party:", avg_dist_original_vs_third)


def find_incorrect_predictions(original, correct):
    return np.nonzero(notequal(original, correct))[0]
    

print("\nDirect copy check with class predictions as outputs:")
detect_direct_copy(*outputs_list)
print("\nDirect copy check with logits as outputs:")
detect_direct_copy(*logits_list)

print("\nAnalyze logit similarity:")
find_logits_similarity(*logits_list)

print("\nDirect copy check with class predictions as outputs (only original's incorrect predictions):")
incorrect_idx = find_incorrect_predictions(outputs_list[0], outputs_list[-1])
print(f"Original model has {len(incorrect_idx)} incorrect predictinos out of {outputs_list[0].shape[0]}.")
detect_direct_copy(*list(map(lambda x: x[incorrect_idx], outputs_list)))

print("\nDirect copy check with logits as outputs (only original's incorrect predictions):")
detect_direct_copy(*list(map(lambda x: x[incorrect_idx], logits_list)))

print("\nAnalyze logit similarity (only original's incorrect predictions):")
find_logits_similarity(*list(map(lambda x: x[incorrect_idx], logits_list)))
                      



