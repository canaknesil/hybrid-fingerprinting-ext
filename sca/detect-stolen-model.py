import sys
import numpy as np
import scipy.stats as st
import scipy.signal as sig
import matplotlib.pyplot as plt
from tqdm import tqdm


#
# PROCESS COMMAND LINE INPUTS
#

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

outputs_original, outputs_suspect, outputs_third, correct_outputs = outputs_list
logits_original, logits_suspect, logits_third, logits_correct = logits_list


traces_original = np.load(original_prefix + "_traces.npy")
traces_suspect = np.load(suspect_prefix + "_traces.npy")
traces_third = np.load(third_prefix + "_traces.npy")

print("traces_original.shape:", traces_original.shape)
print("traces_suspect.shape:", traces_suspect.shape)
print("traces_third.shape:", traces_third.shape)

# Multi-scale representation
dtraces_original = [traces_original]
dtraces_suspect = [traces_suspect]
dtraces_third = [traces_third]

for i in range(4):
    dtraces_original.append(sig.decimate(dtraces_original[-1], 2))
    dtraces_suspect.append(sig.decimate(dtraces_suspect[-1], 2))
    dtraces_third.append(sig.decimate(dtraces_third[-1], 2))


#
# UTILS
#

# Result is in range [0, 2]
def logit_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))


def logits_distance(a, b):
    dist = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        dist[i] = logit_distance(a[i], b[i])
    return dist


def print_histogram(v, *args, **kwargs):
    hist, bin_edges = np.histogram(v, *args, **kwargs)
    print(f"Range:", (bin_edges[0], bin_edges[-1]))
    print("Histogram:", hist)


def one_sample_ttest(a, mean_b):
    return (np.average(a) - mean_b) / (np.std(a) / np.sqrt(len(a)))


def ttest(a, b):
    return st.ttest_ind(a, b).statistic


def pdf(samples, x=None):
    if x is None:
        x = np.linspace(samples.min(), samples.max(), 100)
        
    kde = st.gaussian_kde(samples)
    return x, kde(x)


def pdf_overlap_area(pdf_a, pdf_b, dx=1):
    return np.sum(np.minimum(pdf_a, pdf_b)) * dx


def overlap_between_traces(traces_a, traces_b):
    overlap = np.zeros(traces_a.shape[-1])
    for i in tqdm(range(traces_a.shape[-1])):
        a = traces_a[:,i]
        b = traces_b[:,i]
        min_x = min(list(map(np.min, [a, b])))
        max_x = max(list(map(np.max, [a, b])))
        x = np.linspace(min_x, max_x, 50)
        dx = x[1] - x[0]
        _, pdf_a = pdf(a, x)
        _, pdf_b = pdf(b, x)
        overlap[i] = pdf_overlap_area(pdf_a, pdf_b, dx=dx)
    return overlap


#
# MAIN
#

print("\nAnalyzing class predictions.")

# indices where the original and the 3rd party model have different class predictions
diff_original_vs_third_idx = np.nonzero(outputs_original != outputs_third)[0]
print(f"{len(diff_original_vs_third_idx)} out of {len(outputs_original)} ({len(diff_original_vs_third_idx) / len(outputs_original) * 100}%) class predictions of the original model differs from the 3rd-party model.")

diff_original_vs_suspect_idx = np.nonzero(outputs_original[diff_original_vs_third_idx] != outputs_suspect[diff_original_vs_third_idx])[0]
print(f"For the inputs that produce different prediction by the original and the 3rd-party models, {len(diff_original_vs_suspect_idx)} out of {len(diff_original_vs_third_idx)} ({len(diff_original_vs_suspect_idx) / len(diff_original_vs_third_idx) * 100}%) class predictions of the original model differs from the suspect model.")


print("\nAnalyzing logits.")

diff_original_vs_third = logits_distance(logits_original, logits_third)
print("Histogram of logit distance between the original and 3rd-party models:")
print_histogram(diff_original_vs_third, range=(0, 2))

diff_original_vs_suspect = logits_distance(logits_original, logits_suspect)
print("Histogram of logit distance between the original and suspect models:")
print_histogram(diff_original_vs_suspect, range=(0, 2))

logit_dist_threshold = 1
diff_original_vs_third_idx = np.nonzero(diff_original_vs_third > logit_dist_threshold)[0]
print(f"{len(diff_original_vs_third_idx)} out of {len(outputs_original)} ({len(diff_original_vs_third_idx) / len(outputs_original) * 100}%) logits of the original model has at least {logit_dist_threshold} distant from the 3rd-party model.")

diff_original_vs_suspect = logits_distance(logits_original[diff_original_vs_third_idx], logits_suspect[diff_original_vs_third_idx])
print("For these inputs, logit distance between the original and suspect models:")
print_histogram(diff_original_vs_suspect, range=(0, 2))


print("\nAnalyzing traces.")

scale = 3

print("Calculating overlap between original and 3rd-party models:")
overlap_original_vs_third = overlap_between_traces(dtraces_original[scale], dtraces_third[scale])
print("Calculating overlap between original and suspect models:")
overlap_original_vs_suspect = overlap_between_traces(dtraces_original[scale], dtraces_suspect[scale])

plt.figure()
plt.plot(overlap_original_vs_third)
plt.plot(overlap_original_vs_suspect)
plt.ylim(-1, 1)
    
plt.figure()
plt.plot(overlap_original_vs_suspect - overlap_original_vs_third)
plt.ylim(-1, 1)
plt.title("PDF overlap increase from 3rd-party to suspect")

print("Overlap area original vs 3rd-party:")
print_histogram(overlap_original_vs_third, range=(0, 1))
print("Overlap area original vs suspect:")
print_histogram(overlap_original_vs_suspect, range=(0, 1))

print("Average overlap area between original vs. 3rd-party:", np.average(overlap_original_vs_third))
print("Average overlap area between original vs. suspect  :", np.average(overlap_original_vs_suspect))


# Find indices with less overlap between original vs. 3rd-party
overlap_threshold = 1 # Distance from mean where two Gaussian distribution cross (in terms of standard deviation)
overlap_area_threshold = st.norm.cdf(-overlap_threshold) * 2 # in range [0, 1]
print("overlap_area_threshold:", overlap_area_threshold)

significant_trace_points = np.nonzero(overlap_original_vs_third < overlap_area_threshold)[0]
print(f"{len(significant_trace_points)} trace points are significant.")
print_histogram(significant_trace_points)


# Find indices of inputs that produce less overlap between the original and 3rd-party model.
diff_original_vs_third = np.sum(np.abs(dtraces_original[scale] - dtraces_third[scale]), axis=1)
print("Sum of absolute differences between original and 3rd-party models across trace points:")
print_histogram(diff_original_vs_third)

trace_diff_threshold = (diff_original_vs_third.max() + diff_original_vs_third.min()) / 2
print("trace_diff_threshold:", trace_diff_threshold)
diff_original_vs_third_idx = np.nonzero(diff_original_vs_third > trace_diff_threshold)[0]


# Calculate overlap between the original model and the suspect using only significant points.
print("\nUsing only significant indices.")
print("Calculating overlap between original and 3rd-party models:")
a = dtraces_original[scale][diff_original_vs_third_idx][:,significant_trace_points]
b = dtraces_suspect[scale][diff_original_vs_third_idx][:,significant_trace_points]
c = dtraces_third[scale][diff_original_vs_third_idx][:,significant_trace_points]

overlap_original_vs_third = overlap_between_traces(a, c)
print("Calculating overlap between original and suspect models:")
overlap_original_vs_suspect = overlap_between_traces(a, b)

print("Overlap area original vs 3rd-party:")
print_histogram(overlap_original_vs_third, range=(0, 1))
print("Overlap area original vs suspect:")
print_histogram(overlap_original_vs_suspect, range=(0, 1))

print("Average overlap area between original vs. 3rd-party:", np.average(overlap_original_vs_third))
print("Average overlap area between original vs. suspect  :", np.average(overlap_original_vs_suspect))


#plt.show()
sys.exit()



overlap_original_vs_third = np.zeros(dtraces_original[scale].shape[-1])
overlap_original_vs_suspect = np.zeros(dtraces_original[scale].shape[-1])

print("Calculating PDF overlaps for all trace points:")
for i in tqdm(range(dtraces_original[scale].shape[-1])):
#for i in tqdm(range(2000)): # fast execution for debugging
    a = dtraces_original[scale][:,i]
    b = dtraces_suspect[scale][:,i]
    c = dtraces_third[scale][:,i]
    min_x = min(list(map(np.min, [a, b, c])))
    max_x = max(list(map(np.max, [a, b, c])))
    x = np.linspace(min_x, max_x, 50)
    dx = x[1] - x[0]
    _, pdf_a = pdf(a, x)
    _, pdf_b = pdf(b, x)
    _, pdf_c = pdf(c, x)

    overlap_original_vs_third[i] = pdf_overlap_area(pdf_a, pdf_c, dx=dx)
    overlap_original_vs_suspect[i] = pdf_overlap_area(pdf_a, pdf_b, dx=dx)


plt.figure()
plt.plot(overlap_original_vs_third)
plt.plot(overlap_original_vs_suspect)
plt.ylim(-1, 1)
    
plt.figure()
plt.plot(overlap_original_vs_suspect - overlap_original_vs_third)
plt.ylim(-1, 1)
plt.title("PDF overlap increase from 3rd-party to suspect")

    
#overlap_threshold = 1 # Distance from mean where two Gaussian distribution cross (in terms of standard deviation)
#overlap_area_threshold = st.norm.cdf(-overlap_threshold) * 2 # in range [0, 1]
#print("overlap_area_threshold:", overlap_area_threshold)

print("Overlap area original vs 3rd-party:")
print_histogram(overlap_original_vs_third, range=(0, 1))
print("Overlap area original vs suspect:")
print_histogram(overlap_original_vs_suspect, range=(0, 1))

print("Average overlap area between original vs. 3rd-party:", np.average(overlap_original_vs_third))
print("Average overlap area between original vs. suspect  :", np.average(overlap_original_vs_suspect))



plt.show()
sys.exit()


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
                      



