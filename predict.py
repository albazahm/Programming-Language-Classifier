import pandas as pd
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import to_categorical
import argparse
import numpy as np
import string
import pickle
import re
tf.compat.v1.logging.set_verbosity('ERROR')

def prepare(directory='./Snippets'):

    """
    Function to prepare the text file for prediction by applying preprocessing, sequencing and padding.

    Parameters
    ----------
    directory: str, default="./Snippets"
        The directory containing text files for prediction

    Returns
    -------
    filenames: list
        list of strings corresponding to filenames in the directory for prediction
    snippets: list
        list of preprocessed and padded sequences corresponding to the text files in the directory for prediction 
    """
    #load pickled tokenizer
    with open('./model/tokenizer.p', 'rb') as f:
        tokenizer = pickle.load(f)

    #get filenames in directory
    filenames = os.listdir(directory)
    #initialize list to save code snippets
    snippets = list()

    #preprocess code snippets
    for filename in filenames:

        path = os.path.join(
            directory, 
            filename
            )
        with open(path, 'rb') as f:
            x = f.readlines()

        x = [i.decode() for i in x]
        pattern = re.compile(r'$\n')
        x = [pattern.sub('br', i) for i in x]
        x = ' '.join(x)
        snippets.append(x)

    #transform code snippets to sequences using tokenizer
    snippets = tokenizer.texts_to_sequences(snippets)
    #add padding to the sequences to equalize lengths
    snippets = sequence.pad_sequences(
        snippets, 
        maxlen=1024, 
        padding='post'
        )

    return filenames, snippets

def load_model(mode='CPU'):

    """
    Function to load model from architecture json file and weights h5 file

    Parameters
    ----------
    mode: str, default='CPU'
        runtime mode to use in prediction; choice between CPU or GPU

    Returns
    -------
    model: Keras Model Object
        model object constructed from the json architecture file and h5 model weights file
    """

    #load CPU or GPU architecture from json file depending on input
    if mode=='CPU':
        architecture_path = './model/model_architecture_CPU.json'
    else:
        architecture_path = './model/model_architecture_GPU.json'

    with open(architecture_path, 'rb') as f:
        json_file = f.read()
    
    model = tf.keras.models.model_from_json(json_file)
    #load model weights from h5 file
    model.load_weights('./model/model_weights.h5')

    return model

def get_predictions(model, filenames, snippets, top_n=3):

    """
    Function to get predictions for the code snippet sequences using the model provided

    Parameters
    ----------
    model: Keras Model Object
        Model Object to use for predictions
    filenames: list
        list of strings corresponding to the filenames to be predicted
    snippets: list
        list of preprocessed and padded sequences corresponding to the snippets to be predicted
    top_n: int, default=3
        The top n predictions to output for each snippet
    
    Returns
    -------
    output: tuple
        tuple of format (filename, list of strings of length top_n) containing information about the filename and the top_n predictions
        for that file.
    """

    #load pickled encoder
    with open('./model/encoder.p', 'rb') as f:
        encoder = pickle.load(f)
    #initlize list for predictions to output
    output = list()
    #predict using model
    pred = model.predict(snippets)
    #append top_n class predictions for each filename
    for f, p in zip(filenames, pred):
        class_preds = np.argsort(p)[::-1]
        top_n_preds = class_preds[:top_n]
        top_n_preds = [(encoder.classes_[i], f'{np.round(100*p[i], 2)}%') for i in top_n_preds]
        output.append((f, top_n_preds))

    return output

def _get_args():

    """
    Function to load arguments from command line
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--runtime", 
        default='CPU', 
        help='Use CPU or GPU to make predictions', 
        type=str, 
        choices=['CPU', 'GPU']
        )

    parser.add_argument(
        "--top",
        default=3,
        help='The top N predictions to display for each snippet',
        type=int
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = _get_args()
    runtime = args.runtime
    top = args.top
    #get filenames and snippets
    filenames, snippets = prepare()
    #load model
    model = load_model(mode=runtime)
    #get predictions
    predictions = get_predictions(
        model=model, 
        filenames=filenames, 
        snippets=snippets, 
        top_n=top
        )
    #print predictions
    print('\n\nResults:')
    print('------------')
    for i, v in predictions:
        print('Filename: ' +i +'\n' + 'Prediction: ' + str(v))
        print('\n***********')

    



