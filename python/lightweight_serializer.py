import os
import pandas as pd
import tensorflow as tf

def save_weights(model: tf.keras.Model, path: str):
    ''' A lightweight serializer for tf.keras.Model. It only serializes
    the model weights and saves them individually to disk.
    The corresponding weight information, the type of layer the weight
    belongs to and the data type of the weight values, is stored in
    path/config.json file.
    The weights are stored in path/weight_i.txt with i corresponding to
    the weight index in the config file.

    Inputs:
        model: A tf.keras.Model.

        path: A string identifying the path in which the model weights
        are to be saved.
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    #
    weight_config = []
    serialized_weights = []
    #
    for weight in model.weights:
        if 'ResourceVariable' in str(type(weight)):
            weight_config.append([str(type(weight)), str(weight.dtype)])
            serialized_weights.append(tf.io.serialize_tensor(weight))
        elif 'index_lookup' in str(type(weight)):
            weight_config.append([str(type(weight)), 
                                  str(weight.get_tensors()[0].dtype)])
            serialized_weights.append(
                tf.io.serialize_tensor(weight.get_tensors()[0]))
        else:
            print(f'Unrecognized weight: {str(type(weight))}')
    #
    pd.DataFrame(
        weight_config,
        columns=['weight_type', 'dtype']).to_json(path+'/config.json')
    #
    for i in range(len(serialized_weights)):
        tf.io.write_file(path+f'/weight_{i}.txt', serialized_weights[i])
    return None


def load_weights(path: str):
    ''' A lightweight de-serializer to load the weights of a tf.keras.Model
    saved using the companion lightweight serializer.

    Inputs:
        path: The path where the weights of the model are saved.

    Returns:
        The tuple (config, weights) where config is a Pandas DataFrame with
        columns = ['weight_type', 'dtype'] and weights is a list of weights
        which can be directly loaded using model.set_weights(weights)
        for a model with matching architecture and parameters.
    '''
    config = pd.read_json(path+'/config.json')
    weights = []
    for i in config.index:
        weights.append(
            tf.io.parse_tensor(
                tf.io.read_file(path+f'/weight_{i}.txt'),
                out_type=config.dtype.iloc[i].split("'")[1])
        )
    return config, weights