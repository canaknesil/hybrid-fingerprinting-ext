

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


