import numpy as np


tflite_types = {
    0: "kTfLiteNoType",
    1: "kTfLiteFloat32",
    2: "kTfLiteInt32",
    3: "kTfLiteUInt8",
    4: "kTfLiteInt64",
    5: "kTfLiteString",
    6: "kTfLiteBool",
    7: "kTfLiteInt16",
    8: "kTfLiteComplex64",
    9: "kTfLiteInt8",
    10: "kTfLiteFloat16",
    11: "kTfLiteFloat64",
    12: "kTfLiteComplex128",
    13: "kTfLiteUInt64",
    14: "kTfLiteResource",
    15: "kTfLiteVariant",
    16: "kTfLiteUInt32",
    17: "kTfLiteUInt16",
    18: "kTfLiteInt4",
    19: "kTfLiteBFloat16",
}

tflite_type_sizes = {
    #"kTfLiteNoType": 0,
    "kTfLiteFloat32": 4,
    "kTfLiteInt32": 4,
    "kTfLiteUInt8": 1,
    "kTfLiteInt64": 8,
    #"kTfLiteString": 0,
    #"kTfLiteBool": 0,
    "kTfLiteInt16": 2,
    #"kTfLiteComplex64": 0,
    "kTfLiteInt8": 1,
    "kTfLiteFloat16": 2,
    "kTfLiteFloat64": 8,
    #"kTfLiteComplex128": 0,
    "kTfLiteUInt64": 8,
    #"kTfLiteResource": 0,
    #"kTfLiteVariant": 0,
    "kTfLiteUInt32": 4,
    "kTfLiteUInt16": 2,
    #"kTfLiteInt4": 0,
    #"kTfLiteBFloat16": 0,
}

tflite_type_np_equvalents = {
    "kTfLiteFloat32": np.float32,
}


def tflite_type_to_np(t):
    if type(t) == int:
        t = tflite_types[t]
    assert type(t) == str
    return tflite_type_np_equvalents[t]


def size_of_tflite_type(t):
    if type(t) == int:
        t = tflite_types[t]
    assert type(t) == str
    return tflite_type_sizes[t]
