import compare_models
from compare_models import compare_model_outputs, compare_traces, decimate_traces
from glob import glob
import os
import re
import sys
import numpy as np
from tqdm import tqdm


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
models_snr_100 = ls_re("^mnist_7x7_indep-\d+_init-\d+_snr-100$")
models_retrained = ls_re("^mnist_7x7_indep-\d+_init-\d+_retrained_ds-\d+_init-\d+$")


def print_models(info, models):
    print(f"{info} ({len(models)} items):")
    for m in models:
        print("  ", end="")
        print(m)


print_models("models", models)
print_models("models_snr_100", models_snr_100)
print_models("models_retrained", models_retrained)

print()
correct_outputs = workspace + "/mnist_7x7_indep-0_y_test.npy"
print("Correct outputs:", correct_outputs)
correct_outputs = np.load(correct_outputs)

print()
pairs_snr_100 = list(zip(models, models_snr_100))
print_models("Original vs. Noisy pairs", pairs_snr_100)

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


#
# METRICS TO EVALUATE
#

# Inputs that is provided to the model during inference. "regular"
# inputs are similar to the ones the models is trained with. "random"
# inputs are generated randomly and doesn't have a corresponding
# label.
query_types = ["regular", "random"]

# TODO: add "copy" type
extraction_methods = ["snr-100", "retrained", "third"]

# Information whose similarity between the original and the suspect model will be analyzed.
metric_types = ["class_prediction", "logit", "trace_overlap"]

# The method used to calculate overlap between probability
# distributions between traces from the original and the suspect
# model. Trying only kde and histogram. Numerical integral method
# "integral" is very slow, "gaussian" is wrong.
#overlap_methods = ["kde", "histogram"]
overlap_methods = ["histogram"]

# The filtering that will be applied to the above
# information. "when_orig_wrong" compares only the cases where the
# original model's predictions are wrong.
output_filters = ["none", "when_orig_wrong"]


# TODO: Incorporate overlap_methods into metric_types as e.g. "trace_overlap_kde"
# TODO: Incorporate decimation


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


def load_data(prefix):
    outputs = np.load(prefix + "_outputs.npy")
    traces = np.load(prefix + "_traces.npy")

    traces = decimate_traces(traces, 2, 3)
    
    return outputs, traces



results = init_results([query_types, extraction_methods, metric_types, output_filters])
model_pairs_dict = dict(zip(extraction_methods, [pairs_snr_100, pairs_retrained, pairs_third]))

for extraction_method in extraction_methods:
    
    for query_type in query_types:
        print("\nExtraction method:", extraction_method)
        print("Query type:", query_type)
    
        model_pairs = model_pairs_dict[extraction_method]

        # Load outputs and traces
        print("Loading data.")
        models = []
        for p in model_pairs:
            models += p
        models = set(models)

        data = {}
        for m in models:
            prefix = workspace + "/" + m
            if query_type == "random":
                prefix += "_rand-x"
            print(f"  {prefix}")
            data[m] = load_data(prefix)

        print("Comparing models:")
        for m1, m2 in model_pairs:
            print(f"  ({m1}, {m2})")
            m1_outputs, m1_traces = data[m1]
            m2_outputs, m2_traces = data[m2]

            if query_type == "random":
                m1 += "_rand-x"
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



            
#
# ANALYIS OF RESULTS
#

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
        
    return f"{mean:.5f} +- {std:.5f} (out of {size})"



for a in query_types:
    for c in metric_types:
        for d in output_filters:
            print()
            print(f"query_type: {a}, metric_type: {c}, output_filter: {d}")
            orig_vs_snr = results[a]["snr-100"][c][d]
            orig_vs_retrained = results[a]["retrained"][c][d]
            orig_vs_third = results[a]["third"][c][d]

            print("Original vs. Noisy    :", result_to_str(orig_vs_snr))
            print("Original vs. Retrained:", result_to_str(orig_vs_retrained))
            print("Original vs. 3rd-party:", result_to_str(orig_vs_third))
                  
        


# TODO: For copy model, split the traces in 2 and compare to each other.



