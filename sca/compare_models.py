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


import sys
import numpy as np
import scipy.stats as st
import scipy.signal as sig
from scipy.integrate import quad
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


def pdf_overlap_area(samples_a, samples_b):
    min_x = min(list(map(np.min, [samples_a, samples_b])))
    max_x = max(list(map(np.max, [samples_a, samples_b])))
    x = np.linspace(min_x, max_x, 40)
    dx = x[1] - x[0]
    _, pdf_a = pdf(samples_a, x)
    _, pdf_b = pdf(samples_b, x)
    return np.sum(np.minimum(pdf_a, pdf_b)) * dx


def pdf_overlap_area_quad(samples_a, samples_b):
    kde_a = st.gaussian_kde(samples_a)
    kde_b = st.gaussian_kde(samples_b)
    
    def min_pdf(x):
        return np.minimum(kde_a(x), kde_b(x))

    min_x = min(np.min(samples_a), np.min(samples_b))
    max_x = max(np.max(samples_a), np.max(samples_b))

    overlap_area, _ = quad(min_pdf, min_x, max_x)
    return overlap_area


def pdf_overlap_area_hist(samples_a, samples_b):
    min_s = min(np.min(samples_a), np.min(samples_b))
    max_s = max(np.max(samples_a), np.max(samples_b))

    n_bins = 20 # This number influences the absolute result significantly.
    bins = np.linspace(min_s, max_s, n_bins)
    hist_a, _ = np.histogram(samples_a, bins=bins, density=True)
    hist_b, _ = np.histogram(samples_b, bins=bins, density=True)
    
    # Calculate overlap area
    bin_width = bins[1] - bins[0]
    overlap_area = np.sum(np.minimum(hist_a, hist_b)) * bin_width
    return overlap_area


def pdf_overlap_area_gaussian(samples_a, samples_b):
    # Assuming gaussian distributions for the input samples.
    mean_a = np.average(samples_a)
    mean_b = np.average(samples_b)
    std_a = np.std(samples_a)
    std_b = np.std(samples_b)

    # Bhattacharyya distance
    part1 = 0.25 * (mean_a - mean_b) ** 2 / (std_a ** 2 + std_b ** 2)
    part2 = 0.5 * np.log(0.5 * (std_a ** 2 + std_b ** 2) / (std_a * std_b))
    BD = part1 + part2

    # Bhattacharyya coefficient
    BC = np.exp(-BD)
    overlap_area = BC    
    
    return overlap_area
    


def overlap_between_traces(traces_a, traces_b, method="kde", verbose=False):
    overlap = np.zeros(traces_a.shape[-1])

    v_range = range(traces_a.shape[-1])

    if verbose:
        v_range = tqdm(v_range)
    
    for i in v_range:
        a = traces_a[:,i]
        b = traces_b[:,i]

        if method == "kde":
            # Method with KDE (slow)
            overlap[i] = pdf_overlap_area(a, b)
        elif method == "integral":
            # Method with numerical integration (very slow)
            overlap[i] = pdf_overlap_area_quad(a, b)
        elif method == "histogram":
            # Method with histogram (very fast, but not accurate, parametric)
            overlap[i] = pdf_overlap_area_hist(a, b)
        elif method == "gaussian":
            # Method with Bhattacharyya coefficient (very fast,
            # assuming gaussian dists) This method returns very high
            # overlap every almost every time. Maybe distributions are
            # not close to Gaussian.
            overlap[i] = pdf_overlap_area_gaussian(a, b)
        else:
            raise Exception(f"Unsupported overlap area calculation method: {method}")            
        
    return overlap


def read_model_outputs(outputs_original, outputs_suspect, correct_outputs=None, verbose=False):
    if verbose:
        print("outputs_original.shape:", outputs_original.shape)
        print("outputs_suspect.shape:", outputs_suspect.shape)
        if correct_outputs is not None:
            print("correct_outputs.shape:", correct_outputs.shape)

    if verbose:
        print("Removing dimensions of size 1.")
        
    outputs_original = np.squeeze(outputs_original)
    outputs_suspect = np.squeeze(outputs_suspect)
    if correct_outputs is not None:
        correct_outputs = np.squeeze(correct_outputs)
    
    if correct_outputs is not None:
        assert outputs_original.shape[0] <= correct_outputs.shape[0]
        if outputs_original.shape[0] < correct_outputs.shape[0]:
            if verbose:
                print("Warning: Traces were collected from a subset of test data. Choping test data.")
            correct_outputs = correct_outputs[:outputs_original.shape[0],:]

    assert outputs_suspect.shape == outputs_original.shape
    if correct_outputs is not None:
        assert correct_outputs.shape == outputs_original.shape

    if correct_outputs is not None:
        return outputs_original, outputs_suspect, correct_outputs
    else:
        return outputs_original, outputs_suspect, None



#
# Interface
#

def compare_model_outputs(outputs_original, outputs_suspect, correct_outputs=None, verbose=False):
    # PROCESS INPUT

    outputs_original, outputs_suspect, correct_outputs = read_model_outputs(outputs_original, outputs_suspect, correct_outputs, verbose=verbose)
    
    logits_original = outputs_original
    logits_suspect = outputs_suspect
    if correct_outputs is not None:
        logits_correct = correct_outputs

    outputs_original = logits_to_predictions(outputs_original)
    outputs_suspect = logits_to_predictions(outputs_suspect)
    if correct_outputs is not None:
        correct_outputs = logits_to_predictions(correct_outputs)

        
    # ANALYSIS

    results = {}

    if verbose:
        print("\nAnalyzing class predictions.")
    
    diff_original_vs_suspect_idx = np.nonzero(outputs_original != outputs_suspect)[0]
    results["class_predictions"] = 1 - len(diff_original_vs_suspect_idx) / len(outputs_original)

    if verbose:
        print("Class predictions that differ for the original and the suspect model:")
        print(f"  {frac_str(len(diff_original_vs_suspect_idx), len(outputs_original))}")

    if correct_outputs is not None:
        diff_original_vs_correct_idx = np.nonzero(outputs_original != correct_outputs)[0]
        diff_original_vs_suspect_idx = np.nonzero(outputs_original[diff_original_vs_correct_idx] != outputs_suspect[diff_original_vs_correct_idx])[0]
        results["class_predictions_when_orig_wrong"] = 1 - len(diff_original_vs_suspect_idx) / len(diff_original_vs_correct_idx)

        if verbose:
            print("For the cases where the original model's predictions are not correct, class predictions that differ for the original and the suspect model:")
            print(f"  {frac_str(len(diff_original_vs_suspect_idx), len(diff_original_vs_correct_idx))}")
    

    if verbose:
        print("\nAnalyzing logits.")
    
    diff_original_vs_suspect = logits_distance(logits_original, logits_suspect)
    results["logits"] = (2 - np.average(diff_original_vs_suspect)) / 2 # dividing by 2 so that its range is [0, 1]

    if verbose:
        print("Logit distances between the original and suspect model:")
        print_histogram(diff_original_vs_suspect, range=(0, 2))

    if correct_outputs is not None:
        diff_original_vs_suspect = logits_distance(logits_original[diff_original_vs_correct_idx], logits_suspect[diff_original_vs_correct_idx])
        results["logits_when_orig_wrong"] = (2 - np.average(diff_original_vs_suspect)) / 2

        if verbose:
            print("For the cases where the original model's predictions are not correct, logit distances between the original and suspect model:")
            print_histogram(diff_original_vs_suspect, range=(0, 2))
    
    return results




def compare_traces(outputs_original, outputs_suspect, correct_outputs, traces_original, traces_suspect, overlap_method="kde", verbose=False, plot=False):
    # PROCESS COMMAND LINE INPUTS

    # Model predictions (outputs)
    outputs_original, outputs_suspect, correct_outputs = read_model_outputs(outputs_original, outputs_suspect, correct_outputs)
    
    outputs_original = logits_to_predictions(outputs_original)
    outputs_suspect = logits_to_predictions(outputs_suspect)
    if correct_outputs is not None:
        correct_outputs = logits_to_predictions(correct_outputs)

        
    # Traces
    if verbose:
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
    
    
    # ANALYSIS

    results = {}
    
    if correct_outputs is not None:
        diff_original_vs_correct_idx = np.nonzero(outputs_original != correct_outputs)[0]

    if verbose:
        print("\nAnalyzing traces.")

    if verbose:
        print("Calculating overlap between original and suspect model:")
    overlap_original_vs_suspect = overlap_between_traces(traces_original, traces_suspect, method=overlap_method, verbose=verbose)
    results["trace_overlap"] = np.average(overlap_original_vs_suspect)

    if verbose:
        print("Overlap area original vs suspect:")
        print_histogram(overlap_original_vs_suspect, range=(0, 1))

    if plot:
        plt.figure()
        plt.plot(overlap_original_vs_suspect)
        plt.title("Overlap of Original and Suspect")
        #plt.ylim(0, 1)
    
        ttest_original_vs_suspect = ttest(traces_original, traces_suspect)
        
        plt.figure()
        plt.plot(ttest_original_vs_suspect)
        plt.title("T-test between Original and Suspect")

    if verbose:
        print("Correlation between original and suspect model:")

    corr_original_vs_suspect = correlation(traces_original, traces_suspect)
    
    if plot:
        plt.figure()
        plt.plot(corr_original_vs_suspect)
        plt.title("Correlation between original and suspect model")
        plt.ylim(-1, 1)

    flat_corr = flat_correlation(traces_original, traces_suspect)

    if verbose:
        print("Correlation between flattened versions of original and suspect model:", flat_corr)
    

    if correct_outputs is not None:
        if verbose:
            print("\nUsing only inputs where the original model produce incorrect predictions, together with significant trace points.")
        a = traces_original[diff_original_vs_correct_idx]
        b = traces_suspect[diff_original_vs_correct_idx]

        if verbose:
            print("Calculating overlap between original and suspect model:")

        overlap_original_vs_suspect = overlap_between_traces(a, b, method=overlap_method, verbose=verbose)
        results["trace_overlap_when_orig_wrong"] = np.average(overlap_original_vs_suspect)

        if verbose:
            print("Overlap area original vs suspect:")
            print_histogram(overlap_original_vs_suspect, range=(0, 1))

    return results



def decimate_traces(traces, decimation_factor=2, n_decimation=3):
    for i in range(n_decimation):
        traces = sig.decimate(traces, decimation_factor)
    return traces
    

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

    outputs_original = np.load(original_prefix + "_outputs.npy")
    outputs_suspect = np.load(suspect_prefix + "_outputs.npy")
    correct_outputs = np.load(correct_outputs_file)

    traces_original = np.load(original_prefix + "_traces.npy")
    traces_suspect = np.load(suspect_prefix + "_traces.npy")

    decimation_factor = 2
    n_decimation = 3
    print("decimation:", decimation_factor ** n_decimation)

    traces_original = decimate_traces(traces_original, decimation_factor, n_decimation)
    traces_suspect = decimate_traces(traces_suspect, decimation_factor, n_decimation)    

    results = compare_model_outputs(outputs_original, outputs_suspect, correct_outputs, verbose=True)
    results_2 = compare_traces(outputs_original, outputs_suspect, correct_outputs, traces_original, traces_suspect, verbose=True, plot=True)

    results = {**results, **results_2}

    print("\nSummary:")
    print(results)
    
    plt.show()
    
    
    
    
