import sys
import numpy as np
import scipy.stats as st
import scipy.signal as sig
import matplotlib.pyplot as plt
from tqdm import tqdm


original_prefix = sys.argv[1]
suspect_prefix = sys.argv[2]
third_prefix = sys.argv[3]
correct_outputs_file = sys.argv[4]

print("original_prefix:", original_prefix)
print("suspect_prefix:", suspect_prefix)
print("third_prefix:", third_prefix)
print("correct_outputs_file:", correct_outputs_file)


#
# UTILS
#

def logits_to_predictions(logits):
    assert len(logits.shape) == 2
    predictions = np.zeros(logits.shape[0], dtype=int)
    for i in range(logits.shape[0]):
        predictions[i] = np.argmax(logits[i])
    return predictions


def frac_str(x, y):
    ratio = x / y
    return f"{x} out of {y} ({ratio:.4f})"


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
    print(f"  Range:", (bin_edges[0], bin_edges[-1]))
    print("  Histogram:", hist)
    print("  Average:", np.average(v))
    print("  Standard deviation:", np.std(v))


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
# PROCESS COMMAND LINE INPUTS
#

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

# Multi-scale representation (decimated traces)

# Decimation reduces processing time. But it may also increase
# detection performance. Ideally repeat detection on a multi-scale
# representation of the traces.

# dtraces_original = [traces_original]
# dtraces_suspect = [traces_suspect]
# dtraces_third = [traces_third]

# for i in range(4):
#     dtraces_original.append(sig.decimate(dtraces_original[-1], 2))
#     dtraces_suspect.append(sig.decimate(dtraces_suspect[-1], 2))
#     dtraces_third.append(sig.decimate(dtraces_third[-1], 2))

decimation_factor = 2
n_decimation = 4
print("decimation:", decimation_factor * n_decimation)

for i in range(n_decimation):
    traces_original = sig.decimate(traces_original, 2)
    traces_suspect = sig.decimate(traces_suspect, 2)
    traces_third = sig.decimate(traces_third, 2)


#
# ANALYSIS
#

print("\nAnalyzing class predictions.")

diff_original_vs_third_idx = np.nonzero(outputs_original != outputs_third)[0]
print("Class predictions that differ for the original and the 3rd-party model:")
print(f"  {frac_str(len(diff_original_vs_third_idx), len(outputs_original))}")

diff_original_vs_suspect_idx = np.nonzero(outputs_original != outputs_suspect)[0]
print("Class predictions that differ for the original and the suspect model:")
print(f"  {frac_str(len(diff_original_vs_suspect_idx), len(outputs_original))}")

diff_original_vs_suspect_idx = np.nonzero(outputs_original[diff_original_vs_third_idx] != outputs_suspect[diff_original_vs_third_idx])[0]
print("For the points where the original model's and the 3rd-party model's predictions differ, class predictions that differ for the original and the suspect model:")
print(f"  {frac_str(len(diff_original_vs_suspect_idx), len(diff_original_vs_third_idx))}")


print("\nAnalyzing logits.")

diff_original_vs_third = logits_distance(logits_original, logits_third)
print("Logit distances between the original and 3rd-party model:")
print_histogram(diff_original_vs_third, range=(0, 2))

diff_original_vs_suspect = logits_distance(logits_original, logits_suspect)
print("Logit distances between the original and suspect model:")
print_histogram(diff_original_vs_suspect, range=(0, 2))

# TODO: Use significance rather than threshold
logit_dist_threshold = 1
diff_original_vs_third_idx = np.nonzero(diff_original_vs_third > logit_dist_threshold)[0]
print(f"Logit distance between the original and 3rd-party model that is larger than {logit_dist_threshold}:")
print(f"  {frac_str(len(diff_original_vs_third_idx), len(outputs_original))}")

diff_original_vs_suspect = logits_distance(logits_original[diff_original_vs_third_idx], logits_suspect[diff_original_vs_third_idx])
print(f"For the points where logit distance between the original and the 3rd-party model are larger than {logit_dist_threshold}, logit distances between the original and suspect model:")
print_histogram(diff_original_vs_suspect, range=(0, 2))


print("\nAnalyzing traces.")

print("Calculating overlap between original and 3rd-party models:")
overlap_original_vs_third = overlap_between_traces(traces_original, traces_third)
print("Calculating overlap between original and suspect models:")
overlap_original_vs_suspect = overlap_between_traces(traces_original, traces_suspect)

print("Overlap area original vs 3rd-party:")
print_histogram(overlap_original_vs_third, range=(0, 1))
print("Overlap area original vs suspect:")
print_histogram(overlap_original_vs_suspect, range=(0, 1))


# Find trace points of significance
overlap_threshold = 0.5 # Distance from mean where two Gaussian distribution cross (in terms of standard deviation)
overlap_area_threshold = st.norm.cdf(-overlap_threshold) * 2 # in range [0, 1]
print("overlap_area_threshold:", overlap_area_threshold)

significant_trace_points = np.nonzero(overlap_original_vs_third < overlap_area_threshold)[0]
print("Significant trace points (produce small overlap between original and 3rd-party models):")
print(f"  {frac_str(len(significant_trace_points), traces_original.shape[-1])}")


trace_point_significance = 1 - overlap_original_vs_third
significance_threshold = 1 - overlap_area_threshold

plt.figure()
plt.plot(trace_point_significance)
plt.title("Significance of trace points")
plt.hlines(significance_threshold, 0, len(trace_point_significance))


# Find inputs of significance
diff_original_vs_third = np.sum(np.abs(traces_original - traces_third) * trace_point_significance, axis=1)

#print("Significance of inputs:")
#print_histogram(diff_original_vs_third)


#trace_diff_threshold = np.average(diff_original_vs_third)
top_percentile = 20
print(f"Taking top {top_percentile}% of the inputs that produced the largest difference in traces.")
trace_diff_threshold = np.percentile(diff_original_vs_third, 100 - top_percentile)
#print("Input significance threshold:", trace_diff_threshold)

significant_input_idx = np.nonzero(diff_original_vs_third > trace_diff_threshold)[0]

# We already know the value as we use percentile above.
#print("Significant inputs:")
#print(f"  {frac_str(len(significant_input_idx), traces_original.shape[0])}")


print("\nUsing only significant trace points.")
a = traces_original[significant_input_idx][:,significant_trace_points]
b = traces_suspect[significant_input_idx][:,significant_trace_points]
c = traces_third[significant_input_idx][:,significant_trace_points]

print("Calculating overlap between original and 3rd-party models:")
overlap_original_vs_third = overlap_between_traces(a, c)
print("Calculating overlap between original and suspect models:")
overlap_original_vs_suspect = overlap_between_traces(a, b)

print("Overlap area original vs 3rd-party:")
print_histogram(overlap_original_vs_third, range=(0, 1))
print("Overlap area original vs suspect:")
print_histogram(overlap_original_vs_suspect, range=(0, 1))



plt.show()



