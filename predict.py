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
from sklearn.preprocessing import LabelEncoder
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
    
    TOKENIZER_PATH = './model/tokenizer.p'

    #load pickled tokenizer
    with open(TOKENIZER_PATH, 'rb') as f:
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

        #decode
        x = [i.decode() for i in x]
        #replace end-of-line \n character with <br>
        pattern = re.compile(r'$\n')
        x = [pattern.sub('<br>', i) for i in x]
        #join list
        x = ' '.join(x)
        #replace <br> with empty string
        x = x.replace('<br>', '')
        #replace space with empty string
        x = x.replace(' ', '')
        #append to list
        snippets.append(x)

    #transform code snippets to sequences using tokenizer
    snippets = tokenizer.texts_to_sequences(snippets)
    #add padding to the sequences to equalize lengths
    snippets = sequence.pad_sequences(
        sequences=snippets, 
        maxlen=512, 
        padding='pre',
        truncating='post',
        )

    return filenames, snippets

def load_model():

    """
    Function to load model from architecture json file and weights h5 file

    Returns
    -------
    model: Keras Model Object
        model object constructed from the json architecture file and h5 model weights file
    """

    ARCHITECTURE_PATH = './model/model_architecture.json'
    WEIGHTS_PATH = './model/model_weights.h5'

    #load architecture from json file
    with open(ARCHITECTURE_PATH, 'rb') as f:
        json_file = f.read()
    
    model = tf.keras.models.model_from_json(json_file)
    #load model weights from h5 file
    model.load_weights(WEIGHTS_PATH)

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

    ENCODER_PATH = './model/encoder.p'

    #load pickled encoder
    with open(ENCODER_PATH, 'rb') as f:
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
        "--top",
        default=3,
        help='The top N predictions to display for each snippet',
        type=int
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = _get_args()
    top = args.top
    #get filenames and snippets
    filenames, snippets = prepare()
    #load model
    model = load_model()
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

    



