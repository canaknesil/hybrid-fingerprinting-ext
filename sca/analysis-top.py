import compare_models
from glob import glob
import os
import re
import sys


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


# TODO: Load correct output file
# TODO: Information tuple output from compare_model
# TODO: Pandas table views of the collected and processed info.
# TODO: Try different overlap area methods.
# TODO: For copy model, split the traces in 2 and compare to each other.


#compare_models(original_prefix, suspect_prefix, correct_outputs_file)

