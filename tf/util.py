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


def remove_trailing_slash(s):
    while s[-1] == '/':
        s = s[:-1]
    return s
        

def save_model(model, model_path_prefix):
    # The prefix should not end with /.
    remove_trailing_slash(model_path_prefix)
        
    # Saving model in SavedModel format. This format is recommended for
    # conversion to tflite.
    model_file_savedmodel = model_path_prefix
    print(f'Saving model to {model_file_savedmodel}')
    model.export(model_file_savedmodel)
    
    # SavedModel format is not supported by Keras 3. Saving model also in
    # .keras format for conveniance.
    model_file_keras = model_path_prefix + ".keras"
    print(f'Saving model to {model_file_keras}')
    model.save(model_file_keras)


