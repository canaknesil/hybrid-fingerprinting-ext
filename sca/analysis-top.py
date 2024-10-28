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


import compare_models
from compare_models import compare_model_outputs, compare_traces, decimate_traces
from glob import glob
import os
import re
import sys
import numpy as np
from tqdm import tqdm
import scipy.stats as st
import math
import copy
from functools import reduce


#
# PREPARE MODELS TO BE COMPARED
#

workspace = "workspace"

# With help from ChatGPT
def natural_sort_key(s):
    # Split the string into a list of integers and non-integer parts
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def ls_re(pattern):
    items = os.listdir(workspace)
    items = [i for i in items if re.search(pattern, i)]
    return sorted(items, key=natural_sort_key)


models = ls_re("^mnist_7x7_indep-\d+_init-\d+$")

# There is actually no model whose name ends with "_copy" because they
# would be the same as their equivalent, without the "_copy"
# postfix. However, a second set of traces are collected for these
# models. The name of the trace files end with "_copy_traces.npy".
# models_copy = list(map(lambda s: s + "_copy", models))
models_copy1 = list(map(lambda s: s + "_copy1", models))
models_copy2 = list(map(lambda s: s + "_copy2", models))

models_snr_1000 = ls_re("^mnist_7x7_indep-\d+_init-\d+_snr-1000$")
models_snr_100 = ls_re("^mnist_7x7_indep-\d+_init-\d+_snr-100$")
models_snr_10 = ls_re("^mnist_7x7_indep-\d+_init-\d+_snr-10$")
models_retrained = ls_re("^mnist_7x7_indep-\d+_init-\d+_retrained_ds-\d+_init-\d+$")


def print_models(info, models):
    print(f"{info} ({len(models)} items):")
    for m in models:
        print("  ", end="")
        print(m)


print_models("models", models)
#print_models("models_copy", models_copy)
print_models("models_copy1", models_copy1)
print_models("models_copy2", models_copy2)
print_models("models_snr_1000", models_snr_1000)
print_models("models_snr_100", models_snr_100)
print_models("models_snr_10", models_snr_10)
print_models("models_retrained", models_retrained)

print()
correct_outputs = workspace + "/mnist_7x7_indep-0_y_test.npy"
print("Correct outputs:", correct_outputs)
correct_outputs = np.load(correct_outputs)

print()
#pairs_copy = list(zip(models, models_copy))
pairs_copy = list(zip(models_copy1, models_copy2))
pairs_snr_1000 = list(zip(models, models_snr_1000))
pairs_snr_100 = list(zip(models, models_snr_100))
pairs_snr_10 = list(zip(models, models_snr_10))
print_models("Original vs. Copy pairs", pairs_copy)
print_models("Original vs. Noisy (SNR=1000) pairs", pairs_snr_1000)
print_models("Original vs. Noisy (SNR=100) pairs", pairs_snr_100)
print_models("Original vs. Noisy (SNR=10) pairs", pairs_snr_10)

orig_models = []
for i in range(len(models)):
    if i % 4 == 0 or i % 4 == 1:
        for j in range(2):
            orig_models.append(models[i])

pairs_retrained = list(zip(orig_models, models_retrained))
print_models("Original vs. Retrained pairs", pairs_retrained)

n = len(models)
pairs_third = list(zip([models[0]] * (n-2), models[2:]))
print_models("Original vs. 3rd-party pairs", pairs_third)


debug = False
if debug:
    n = 3
    pairs_copy = pairs_copy[:n]
    pairs_snr_1000 = pairs_snr_1000[:n]
    pairs_snr_100 = pairs_snr_100[:n]
    pairs_snr_10 = pairs_snr_10[:n]
    pairs_retrained = pairs_retrained[:n]
    pairs_third = pairs_third[:n]
    

#sys.exit()


#
# METRICS TO EVALUATE
#

# Inputs that is provided to the model during inference. "regular"
# inputs are similar to the ones the models is trained with. "random"
# inputs are generated randomly and doesn't have a corresponding
# label.
#query_types = ["regular", "random"]
query_types = ["regular"]

extraction_methods = ["copy", "snr-1000", "snr-100", "snr-10", "retrained", "third"]

# Information whose similarity between the original and the suspect model will be analyzed.
metric_types = ["class_prediction", "logit", "trace_overlap"]

# The method used to calculate overlap between probability
# distributions between traces from the original and the suspect
# model. Trying only kde and histogram. Numerical integral method
# "integral" is very slow, "gaussian" is wrong. The variable
# overlap_methods is not currently in use. It may be incorporated into
# the metric_types variable.
#overlap_methods = ["kde", "histogram"]

# The filtering that will be applied to the above
# information. "when_orig_wrong" compares only the cases where the
# original model's predictions are wrong.
#output_filters = ["none", "when_orig_wrong"]
output_filters = ["none"]


#
# COMPARE MODELS
#

def init_results(dims):
    if len(dims) == 0:
        return []
    else:
        res = {}
        for x in dims[0]:
            res[x] = init_results(dims[1:])
        return res


# Loading traces takes time. Using a cache to prevent reloading over
# and over again. Eviction is not implemented.
load_cache = {}
def load_data(prefix):
    if prefix in load_cache:
        return load_cache[prefix]
    else:
        data = load_data_from_disk(prefix)
        load_cache[prefix] = data
        return data

    
def load_data_from_disk(prefix):
    outputs = np.load(prefix + "_outputs.npy")
    traces = np.load(prefix + "_traces.npy") # Using mmap_mode isn't faster.

    traces = decimate_traces(traces, 2, 2) # (factor=2, n_decimation=2) is found to be a good compromise.
    
    return outputs, traces


results = init_results([query_types, extraction_methods, metric_types, output_filters])
model_pairs_dict = dict(zip(extraction_methods, [pairs_copy, pairs_snr_1000, pairs_snr_100, pairs_snr_10, pairs_retrained, pairs_third]))

for extraction_method in extraction_methods:
    
    for query_type in query_types:
        print("\nExtraction method:", extraction_method)
        print("Query type:", query_type)
    
        model_pairs = model_pairs_dict[extraction_method]

        print("Comparing models:")
        for m1, m2 in model_pairs:
            print(f"  ({m1}, {m2})")

            # Load outputs and traces
            prefix1 = workspace + "/" + m1
            prefix2 = workspace + "/" + m2
            if query_type == "random":
                prefix1 += "_rand-x"
                prefix2 += "_rand-x"

            m1_outputs, m1_traces = load_data(prefix1)
            m2_outputs, m2_traces = load_data(prefix2)
            
            if query_type == "random":
                m1 += "_rand-x"
                corr_outputs = None
            elif "when_orig_wrong" not in output_filters:
                corr_outputs = None
            else:
                corr_outputs = correct_outputs
                
            res = compare_model_outputs(m1_outputs, m2_outputs, corr_outputs)
            results[query_type][extraction_method]["class_prediction"]["none"].append(res["class_predictions"])
            results[query_type][extraction_method]["logit"]["none"].append(res["logits"])

            if corr_outputs is not None:
                results[query_type][extraction_method]["class_prediction"]["when_orig_wrong"].append(res["class_predictions_when_orig_wrong"])
                results[query_type][extraction_method]["logit"]["when_orig_wrong"].append(res["logits_when_orig_wrong"])

            res = compare_traces(m1_outputs, m2_outputs, corr_outputs, m1_traces, m2_traces, overlap_method="histogram")
            results[query_type][extraction_method]["trace_overlap"]["none"].append(res["trace_overlap"])

            if corr_outputs is not None:
                results[query_type][extraction_method]["trace_overlap"]["when_orig_wrong"].append(res["trace_overlap_when_orig_wrong"])


print("\nResults:")
print(results)
                
results_file = workspace + "/analysis-results"
print("Writing results to " + results_file)

with open(results_file, "w") as f:
    print(results, file=f)
    

#
# INTERPRETATION OF RESULTS
#

def pdf_overlap_area_gaussian(mean_a, std_a, mean_b, std_b):
    # Bhattacharyya distance
    part1 = 0.25 * (mean_a - mean_b) ** 2 / (std_a ** 2 + std_b ** 2)
    part2 = 0.5 * np.log(0.5 * (std_a ** 2 + std_b ** 2) / (std_a * std_b))
    BD = part1 + part2

    # Bhattacharyya coefficient
    BC = np.exp(-BD)
    overlap_area = BC    
    
    return overlap_area

def confusion_metrics(samples_a, samples_b):
    # st.gaussian_kde doesn't work when all samples are
    # equal. Defining kde_a as a normal distribution with std
    # 1/sqrt(N) didn't work. Adding a small noise to the samples
    # didn't work either.
    is_identical_a = all([s == samples_a[0] for s in samples_a])
    is_identical_b = all([s == samples_b[0] for s in samples_b])

    if is_identical_a and is_identical_b:
        # Assuming number of samples are the same for a and b.
        # mean_a = samples_a[0]
        # mean_b = samples_b[0]
        # std_a = 1 / math.sqrt(len(samples_a))
        # std_b = 1 / math.sqrt(len(samples_b))
        # overlap_area = pdf_overlap_area_gaussian(mean_a, std_a, mean_b, std_b)
        # tpr = tnr = 1 - overlap_area / 2
        # fpr = fnr = overlap_area / 2
        # return tpr, tnr, fpr, fnr
        return np.nan, np.nan, np.nan, np.nan

    if is_identical_a or is_identical_b:
        # Assuming large number of samples so the distribution of a is
        # very thin and tall.
        # tpr = tnr = 1
        # fpr = fnr = 0
        # return tpr, tnr, fpr, fnr
        return np.nan, np.nan, np.nan, np.nan

    kde_a = st.gaussian_kde(samples_a)
    kde_b = st.gaussian_kde(samples_b)

    min_x = min(list(map(np.min, [samples_a, samples_b])))
    max_x = max(list(map(np.max, [samples_a, samples_b])))
    extra = (max_x - min_x) / 4
    min_x -= extra
    max_x += extra
    
    x = np.linspace(min_x, max_x, 10000000)
    dx = x[1] - x[0]

    pdf_a = kde_a(x)
    pdf_b = kde_b(x)

    tpr = np.sum(pdf_a[pdf_a > pdf_b]) * dx
    tnr = np.sum(pdf_b[pdf_b > pdf_a]) * dx
    fpr = np.sum(pdf_b[pdf_a > pdf_b]) * dx
    fnr = np.sum(pdf_a[pdf_b > pdf_a]) * dx

    return tpr, tnr, fpr, fnr


def confusion_metrics_joint(samples_a, samples_b):
    samples_a = np.array(samples_a)
    samples_b = np.array(samples_b)
    assert len(samples_a.shape) == len(samples_b.shape) == 2 # (dims, data)

    try:
        kde_a = st.gaussian_kde(samples_a)
        kde_b = st.gaussian_kde(samples_b)
    except np.linalg.LinAlgError:
        # Probably samples in one or more sample sets are equal.
        return np.nan, np.nan, np.nan, np.nan

    mins_a = np.min(samples_a, axis=1)
    mins_b = np.min(samples_b, axis=1)
    mins = np.minimum(mins_a, mins_b)
    maxs_a = np.max(samples_a, axis=1)
    maxs_b = np.max(samples_b, axis=1)
    maxs = np.maximum(maxs_a, maxs_b)

    # Expand min and max so that ot covers pdf's tails.
    extra = (maxs - mins) / 4
    mins -= extra
    maxs += extra

    x = []
    for minval, maxval in zip(mins, maxs):
        x.append(np.linspace(minval, maxval, 1000))
    dx = reduce(lambda a, b: a*b, map(lambda a: a[1] - a[0], x))

    x = list(map(lambda a: a.flatten(), np.meshgrid(*x)))

    pdf_a = kde_a(x)
    pdf_b = kde_b(x)

    tpr = np.sum(pdf_a[pdf_a > pdf_b]) * dx
    tnr = np.sum(pdf_b[pdf_b > pdf_a]) * dx
    fpr = np.sum(pdf_b[pdf_a > pdf_b]) * dx
    fnr = np.sum(pdf_a[pdf_b > pdf_a]) * dx

    return tpr, tnr, fpr, fnr


def confusion_metrics_multi(samples_a, *samples_b):
    # Assuming samples in a are not equal.
    # Ignoring sets where samples are equal in b.
    samples_b = list(filter(lambda ss: not all([s == ss[0] for s in ss]), samples_b))
    
    kde_a = st.gaussian_kde(samples_a)
    kde_b = list(map(st.gaussian_kde, samples_b))

    min_x = min(list(map(np.min, [samples_a, *samples_b])))
    max_x = max(list(map(np.max, [samples_a, *samples_b])))
    x = np.linspace(min_x, max_x, 100000)
    dx = x[1] - x[0]

    pdf_a = kde_a(x)
    pdf_b = list(map(lambda f: f(x), kde_b))

    tpr = np.sum(pdf_a[np.all([pdf_a > p for p in pdf_b], axis=0)]) * dx
    fnr = 1 - tpr

    return tpr, np.nan, np.nan, fnr


def confusion_metrics_majority_voting(cm1, cm2):
    tpr1, tnr1, fpr1, fnr1 = cm1
    tpr2, tnr2, fpr2, fnr2 = cm2

    tpr = tpr1 * tpr2
    tnr = tnr1 * tnr2
    fpr = fpr1 * fpr2
    fnr = fnr1 * fnr2
    return tpr, tnr, fpr, fnr


def confusion_metrics_and(cm1, cm2):
    tpr1, tnr1, fpr1, fnr1 = cm1
    tpr2, tnr2, fpr2, fnr2 = cm2

    tpr = tpr1 * tpr2
    tnr = tnr1 * tnr2
    fpr = fpr1 * fpr2
    fnr = 1 - tpr
    return tpr, tnr, fpr, fnr


def confusion_metrics_or(cm1, cm2):
    tpr1, tnr1, fpr1, fnr1 = cm1
    tpr2, tnr2, fpr2, fnr2 = cm2

    tpr = 1 - (1 - tpr1) * (1 - tpr2)
    tnr = tnr1 * tnr2
    fpr = 1 - (1 - fpr1) * (1 - fpr2)
    fnr = fnr1 * fnr2
    return tpr, tnr, fpr, fnr


def cm_to_accuracy(tpr, tnr, fpr, fnr):
    return (tpr + tnr) / 2


def cm_to_f1_score(tpr, tnr, fpr, fnr):
    return 2 * tpr / (tpr + fpr)


def stats_to_str(r):
    r = np.array(r)
    assert(len(r.shape) == 1)
    
    if len(r) == 0:
        mean = float("nan")
        std = float("nan")
        size = 0
    elif len(r) == 1:
        mean = r[0]
        std = float("nan")
        size = 1
    else:
        mean = np.average(r)
        std = np.std(r)
        size = len(r)
        
    return f"{mean:.4f} +- {std:.4f} (out of {size})"


def tuple_to_str(r):
    if len(r) == 0:
        return "()"
    
    s = "("
    s += f"{r[0]:.4f}"
    for x in r[1:]:
        s += f" {x:.4f}"
    s += ")"
    return s


def cm_improvement(cm1, cm2):
    x = np.array(cm1)
    y = np.array(cm2)
    return (y - x)


for a in query_types:
    for c in metric_types:
        for d in output_filters:
            print()
            print(f"query_type: {a}, metric_type: {c}, output_filter: {d}")
                
            orig_vs_third = results[a]["third"][c][d]
            print(f"Original vs. third: {stats_to_str(orig_vs_third)}")
            
            for e in filter(lambda x: x != "third", extraction_methods):
                orig_vs_suspect = results[a][e][c][d]
                orig_vs_suspect_cm = confusion_metrics(orig_vs_suspect, orig_vs_third)
                
                print(f"Original vs. {e}: {stats_to_str(orig_vs_suspect)} (tpr, tnr, fpr, fnr)={tuple_to_str(orig_vs_suspect_cm)}")


for a in query_types:
    for d in output_filters:
        print()
        print(f"query_type: {a}, output_filter: {d}")
            
        orig_vs_third_prediction = results[a]["third"]["class_prediction"][d]
        orig_vs_third_logit = results[a]["third"]["logit"][d]
        orig_vs_third_trace = results[a]["third"]["trace_overlap"][d]

        for e in filter(lambda x: x != "third", extraction_methods):
            orig_vs_suspect_prediction = results[a][e]["class_prediction"][d]
            orig_vs_suspect_logit = results[a][e]["logit"][d]
            orig_vs_suspect_trace = results[a][e]["trace_overlap"][d]

            orig_vs_suspect_prediction_cm = confusion_metrics(orig_vs_suspect_prediction, orig_vs_third_prediction)
            orig_vs_suspect_logit_cm = confusion_metrics(orig_vs_suspect_logit, orig_vs_third_logit)
            orig_vs_suspect_trace_cm = confusion_metrics(orig_vs_suspect_trace, orig_vs_third_trace)

            hybrid_prediction_and_trace_cm = confusion_metrics_joint([orig_vs_suspect_prediction, orig_vs_suspect_trace],
                                                                     [orig_vs_third_prediction, orig_vs_third_trace])
            hybrid_logit_and_trace_cm = confusion_metrics_joint([orig_vs_suspect_logit, orig_vs_suspect_trace],
                                                                [orig_vs_third_logit, orig_vs_third_trace])

            hybrid_prediction_and_trace_improvement = cm_improvement(orig_vs_suspect_prediction_cm, hybrid_prediction_and_trace_cm)
            hybrid_logit_and_trace_improvement = cm_improvement(orig_vs_suspect_logit_cm, hybrid_logit_and_trace_cm)

            print(f"(Hybrid prection + trace) Original vs. {e}: (tpr, tnr, fpr, fnr)={tuple_to_str(hybrid_prediction_and_trace_cm)} improvement={tuple_to_str(hybrid_prediction_and_trace_improvement)}")
            print(f"(Hybrid logit + trace) Original vs. {e}   : (tpr, tnr, fpr, fnr)={tuple_to_str(hybrid_logit_and_trace_cm)} improvement={tuple_to_str(hybrid_logit_and_trace_improvement)}")
            
