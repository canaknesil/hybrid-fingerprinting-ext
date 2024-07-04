import numpy as np
import sys
import matplotlib.pyplot as plt


traces_files = sys.argv[1:]
print("traces_files:")
for f in traces_files:
    print(f)
print()

traces = list(map(np.load, traces_files))
print("shapes:")
for t in traces:
    print(t.shape)
print()

assert len(traces) >= 2
assert len(traces[0].shape) == 2
assert all(t.shape == traces[0].shape for t in traces[1:])

traces = np.array(traces)

# traces [set, trace, trace_point]
# traces [set, observation, variable]


def correlation(traces):
    # traces [set, observation, variable]
    assert len(traces.shape) == 3

    # corr [cor_mat_x, cor_mat_y, variable]
    corr = np.zeros([traces.shape[0], traces.shape[0], traces.shape[2]], dtype=traces.dtype) 
    for v in range(traces.shape[2]):
        corr[:,:,v] = np.corrcoef(traces[:,:,v])

    return corr


def flat_correlation(traces):
    # traces [set, observation, variable]
    assert len(traces.shape) == 3
    
    traces = np.reshape(traces, [traces.shape[0], -1])
    corr = np.corrcoef(traces)
    return corr


corr = correlation(traces)

for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        if i < j:
            plt.figure()
            plt.plot(corr[i,j])
            plt.title(f"Correlation of traces no. {i+1} vs. {j+1}")
            plt.ylim(-1, 1)


flat_corr = flat_correlation(traces)

print("Correlation of flattened trace sets:")
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        if i < j:
            print(f"{i+1} vs. {j+1}: {flat_corr[i, j]}")


#plt.show()

# TODO: Find trace points where correlation differs the most.
# TODO: Multi-scale representation. Gaussian scale-space representation.

