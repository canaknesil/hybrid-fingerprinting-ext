import sys
import numpy as np
import scipy.stats as st
import scipy.signal as sig
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    print("  Nr. or items:", len(v))
    print("  Range:", (bin_edges[0], bin_edges[-1]))
    print("  Histogram:", hist)
    print("  Average:", np.average(v))
    print("  Standard deviation:", np.std(v))


def one_sample_ttest(a, mean_b):
    return (np.average(a) - mean_b) / (np.std(a) / np.sqrt(len(a)))


def ttest(a, b):
    return st.ttest_ind(a, b).statistic


def correlation(traces_a, traces_b):
    # traces [observation, variable]
    corr = np.zeros(traces_a.shape[-1])
    for v in range(traces_a.shape[-1]):
        corr[v] = np.corrcoef(traces_a[:,v], traces_b[:,v])[0,1]
    return corr


def flat_correlation(traces_a, traces_b):
    traces_a = traces_a.flatten()
    traces_b = traces_b.flatten()
    corr = np.corrcoef(traces_a, traces_b)[0,1]
    return corr


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
# Interface
#

def compare_models(original_prefix, suspect_prefix, correct_outputs_file, decimation_factor=2, n_decimation=3):

    # PROCESS COMMAND LINE INPUTS

    outputs_original = np.load(original_prefix + "_outputs.npy")
    outputs_suspect = np.load(suspect_prefix + "_outputs.npy")
    correct_outputs = np.load(correct_outputs_file)
    
    print("outputs_original.shape:", outputs_original.shape)
    print("outputs_suspect.shape:", outputs_suspect.shape)
    print("correct_outputs.shape:", correct_outputs.shape)
    
    print("Removing dimensions of size 1.")
    outputs_original = np.squeeze(outputs_original)
    outputs_suspect = np.squeeze(outputs_suspect)
    correct_outputs = np.squeeze(correct_outputs)
    
    
    assert outputs_original.shape[0] <= correct_outputs.shape[0]
    if outputs_original.shape[0] < correct_outputs.shape[0]:
        print("Warning: Traces were collected from a subset of test data. Choping test data.")
        correct_outputs = correct_outputs[:outputs_original.shape[0],:]
    
    outputs_list = [outputs_original, outputs_suspect, correct_outputs]
    for x in outputs_list[1:]:
        assert x.shape == outputs_list[0].shape
    
    
    logits_list = outputs_list
    outputs_list = list(map(logits_to_predictions, outputs_list))
    
    outputs_original, outputs_suspect, correct_outputs = outputs_list
    logits_original, logits_suspect, logits_correct = logits_list
    
    
    traces_original = np.load(original_prefix + "_traces.npy")
    traces_suspect = np.load(suspect_prefix + "_traces.npy")
    
    print("traces_original.shape:", traces_original.shape)
    print("traces_suspect.shape:", traces_suspect.shape)
    
    #points_start = 0
    #points_stop = 3200
    #print(f"Trimming trace points from [0, {traces_original.shape[-1]}] to [{points_start}, {points_stop}].")
    #traces_original = traces_original[:,points_start:points_stop]
    #traces_suspect = traces_suspect[:,points_start:points_stop]
    
    
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
    
    print("decimation:", decimation_factor ** n_decimation)
    
    for i in range(n_decimation):
        traces_original = sig.decimate(traces_original, 2)
        traces_suspect = sig.decimate(traces_suspect, 2)
    
    
    # ANALYSIS
    
    print("\nAnalyzing class predictions.")
    
    diff_original_vs_suspect_idx = np.nonzero(outputs_original != outputs_suspect)[0]
    print("Class predictions that differ for the original and the suspect model:")
    print(f"  {frac_str(len(diff_original_vs_suspect_idx), len(outputs_original))}")
    
    diff_original_vs_correct_idx = np.nonzero(outputs_original != correct_outputs)[0]
    diff_original_vs_suspect_idx = np.nonzero(outputs_original[diff_original_vs_correct_idx] != outputs_suspect[diff_original_vs_correct_idx])[0]
    print("For the cases where the original model's predictions are not correct, class predictions that differ for the original and the suspect model:")
    print(f"  {frac_str(len(diff_original_vs_suspect_idx), len(diff_original_vs_correct_idx))}")
    
    
    print("\nAnalyzing logits.")
    
    diff_original_vs_suspect = logits_distance(logits_original, logits_suspect)
    print("Logit distances between the original and suspect model:")
    print_histogram(diff_original_vs_suspect, range=(0, 2))
    
    diff_original_vs_suspect = logits_distance(logits_original[diff_original_vs_correct_idx], logits_suspect[diff_original_vs_correct_idx])
    print("For the cases where the original model's predictions are not correct, logit distances between the original and suspect model:")
    print_histogram(diff_original_vs_suspect, range=(0, 2))
    
    
    print("\nAnalyzing traces.")
    
    print("Calculating overlap between original and suspect model:")
    overlap_original_vs_suspect = overlap_between_traces(traces_original, traces_suspect)
    
    print("Overlap area original vs suspect:")
    print_histogram(overlap_original_vs_suspect, range=(0, 1))
    
    plt.figure()
    plt.plot(overlap_original_vs_suspect)
    plt.title("Overlap of Original and Suspect")
    plt.ylim(0, 1)
    
    print("T-test between original and suspect model:")
    ttest_original_vs_suspect = ttest(traces_original, traces_suspect)
    
    plt.figure()
    plt.plot(ttest_original_vs_suspect)
    plt.title("T-test between Original and Suspect")
    
    print("Correlation between original and suspect model:")
    corr_original_vs_suspect = correlation(traces_original, traces_suspect)
    
    plt.figure()
    plt.plot(corr_original_vs_suspect)
    plt.title("Correlation between original and suspect model")
    plt.ylim(-1, 1)
    
    print("Correlation between flattened versions of original and suspect model:", flat_correlation(traces_original, traces_suspect))
    
    
    print("\nUsing only inputs where the original model produce incorrect predictions, together with significant trace points.")
    a = traces_original[diff_original_vs_correct_idx]
    b = traces_suspect[diff_original_vs_correct_idx]
    
    print("Calculating overlap between original and suspect model:")
    overlap_original_vs_suspect = overlap_between_traces(a, b)
    
    print("Overlap area original vs suspect:")
    print_histogram(overlap_original_vs_suspect, range=(0, 1))
    

#
# MAIN
#

if __name__ == '__main__':
    original_prefix = sys.argv[1]
    suspect_prefix = sys.argv[2]
    correct_outputs_file = sys.argv[3]
    
    print("original_prefix:", original_prefix)
    print("suspect_prefix:", suspect_prefix)
    print("correct_outputs_file:", correct_outputs_file)

    compare_models(original_prefix, suspect_prefix, correct_outputs_file)
    
    plt.show()
    
    
    
    
