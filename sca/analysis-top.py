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
        mean_a = samples_a[0]
        mean_b = samples_b[0]
        std_a = 1 / math.sqrt(len(samples_a))
        std_b = 1 / math.sqrt(len(samples_b))
        overlap_area = pdf_overlap_area_gaussian(mean_a, std_a, mean_b, std_b)
        tpr = tnr = 1 - overlap_area / 2
        fpr = fnr = overlap_area / 2
        return tpr, tnr, fpr, fnr

    if is_identical_a or is_identical_b:
        # Assuming large number of samples so the distribution of a is
        # very thin and tall.
        tpr = tnr = 1
        fpr = fnr = 0
        return tpr, tnr, fpr, fnr

    kde_a = st.gaussian_kde(samples_a)
    kde_b = st.gaussian_kde(samples_b)

    min_x = min(list(map(np.min, [samples_a, samples_b])))
    max_x = max(list(map(np.max, [samples_a, samples_b])))
    x = np.linspace(min_x, max_x, 1000000)
    dx = x[1] - x[0]

    pdf_a = kde_a(x)
    pdf_b = kde_b(x)

    tpr = np.sum(pdf_a[pdf_a > pdf_b]) * dx
    tnr = np.sum(pdf_b[pdf_b > pdf_a]) * dx
    fpr = 1 - tnr
    fnr = 1 - tpr

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


def result_to_str(r):
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


# TODO: Joint metric (that uses joint probability distribution).


simple_results = copy.deepcopy(results)

for a in query_types:
    for c in metric_types:
        for d in output_filters:
            print()
            print(f"query_type: {a}, metric_type: {c}, output_filter: {d}")
            orig_vs_copy = results[a]["copy"][c][d]
            orig_vs_snr_1000 = results[a]["snr-1000"][c][d]
            orig_vs_snr_100 = results[a]["snr-100"][c][d]
            orig_vs_snr_10 = results[a]["snr-10"][c][d]
            orig_vs_retrained = results[a]["retrained"][c][d]
            orig_vs_third = results[a]["third"][c][d]

            orig_vs_copy_cm = confusion_metrics(orig_vs_copy, orig_vs_third)
            orig_vs_snr_1000_cm = confusion_metrics(orig_vs_snr_1000, orig_vs_third)
            orig_vs_snr_100_cm = confusion_metrics(orig_vs_snr_100, orig_vs_third)
            orig_vs_snr_10_cm = confusion_metrics(orig_vs_snr_10, orig_vs_third)
            orig_vs_retrained_cm = confusion_metrics(orig_vs_retrained, orig_vs_third)
            orig_vs_third_cm = confusion_metrics_multi(orig_vs_third, orig_vs_copy, orig_vs_snr_1000, orig_vs_snr_100, orig_vs_snr_10, orig_vs_retrained)

            print(f"Original vs. Copy            : {result_to_str(orig_vs_copy)     } (tpr, tnr, fpr, fnr)={tuple_to_str(orig_vs_copy_cm)}")
            print(f"Original vs. Noisy (SNR=1000): {result_to_str(orig_vs_snr_1000) } (tpr, tnr, fpr, fnr)={tuple_to_str(orig_vs_snr_1000_cm)}")
            print(f"Original vs. Noisy (SNR=100) : {result_to_str(orig_vs_snr_100)  } (tpr, tnr, fpr, fnr)={tuple_to_str(orig_vs_snr_100_cm)}")
            print(f"Original vs. Noisy (SNR=10)  : {result_to_str(orig_vs_snr_10)   } (tpr, tnr, fpr, fnr)={tuple_to_str(orig_vs_snr_10_cm)}")
            print(f"Original vs. Retrained       : {result_to_str(orig_vs_retrained)} (tpr, tnr, fpr, fnr)={tuple_to_str(orig_vs_retrained_cm)}")
            print(f"Original vs. 3rd-party       : {result_to_str(orig_vs_third)    } (tpr, tnr, fpr, fnr)={tuple_to_str(orig_vs_third_cm)}")
            
            simple_results[a]["copy"][c][d] = orig_vs_copy_cm
            simple_results[a]["snr-1000"][c][d] = orig_vs_snr_1000_cm
            simple_results[a]["snr-100"][c][d] = orig_vs_snr_100_cm
            simple_results[a]["snr-10"][c][d] = orig_vs_snr_10_cm
            simple_results[a]["retrained"][c][d] = orig_vs_retrained_cm
            simple_results[a]["third"][c][d] = orig_vs_third_cm


for a in query_types:
    for d in output_filters:
        for e in extraction_methods:
            class_prediction_cm = simple_results[a][e]["class_prediction"][d]
            logit_cm = simple_results[a][e]["logit"][d]
            trace_overlap_cm = simple_results[a][e]["trace_overlap"][d]

            accuracy_class_prediction = cm_to_accuracy(*class_prediction_cm)
            accuracy_logit = cm_to_accuracy(*logit_cm)

            f1_class_prediction = cm_to_f1_score(*class_prediction_cm)
            f1_logit = cm_to_f1_score(*logit_cm)
            
            hybrid_methods = {"AND": confusion_metrics_and,
                              "OR" : confusion_metrics_or}
            
            for m, merge_cm in hybrid_methods.items():
                print()
                print(f"query_type: {a}, output_filter: {d}, extraction_method: {e}, hybrid method: {m}")

                hybrid1_cm = merge_cm(class_prediction_cm, trace_overlap_cm)
                hybrid2_cm = merge_cm(logit_cm, trace_overlap_cm)
                
                accuracy_hybrid1 = cm_to_accuracy(*hybrid1_cm)
                accuracy_hybrid2 = cm_to_accuracy(*hybrid2_cm)

                f1_hybrid1 = cm_to_f1_score(*hybrid1_cm)
                f1_hybrid2 = cm_to_f1_score(*hybrid2_cm)            
                
                improvement1 = (f1_hybrid1 - f1_class_prediction) / f1_class_prediction
                improvement2 = (f1_hybrid2 - f1_logit) / f1_logit
                
                print(f"class_prediction                       : accuracy={f1_class_prediction:.4f}")
                print(f"Hybrid class_prediction + trace_overlap: accuracy={f1_hybrid1:.4f}")
                print(f"                                      improvement={improvement1:.4f}")
                print(f"logit                                  : accuracy={f1_logit:.4f}")
                print(f"Hybrid logit            + trace_overlap: accuracy={f1_hybrid2:.4f}")
                print(f"                                      improvement={improvement2:.4f}")

            
