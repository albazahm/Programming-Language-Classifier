import pandas as pd
import logging
import os
from sklearn.preprocessing import LabelEncoder
import argparse
import numpy as np
import string
import pickle
import re
import time

def identity_tokenizer(text):
    return text

def preprocess(text):
    """
    Function to preprocess text, removing non-english, non-numeric, multiple spaces, and html before tokenizationation.

    Parameters
    ----------
    text: str
      a row of text to preprocess

    Returns
    -------
    text: str
      a row of preprocessed text
    """
    #cast row as string
    text = str(text)
    #replace complex string for comma (complex string was used to avoid csv interpretting commas as new line)
    text = text.replace('!@#$%^&&^%$#@!', ',')
    #remove characters that are non-english, non-numeric or not a space
    text = ''.join([t for t in text if t in string.printable or t==' '])
    #wrap all punctuation by space
    text = re.sub(fr'([{string.punctuation}])', r' \1 ', text)
    #replace any instance of 2 or more spaces with one space
    text = re.sub(r' {2,}', ' ', text)
    #replace br html with newline character
    text = text.replace(' br ', ' \n ')
    #split text on spaces
    text = text.split(' ')
    #remove empty strings
    text = [t for t in text if t!='']
    
    return text

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
    num_files: int
        number of files with .txt extension
    """
    
    VECTORIZER_PATH = './model/vectorizer.p'

    #load pickled tokenizer
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)

    #get filenames in directory
    filenames = os.listdir(directory)
    #isolate files with .txt extension
    filenames = [f for f in filenames if f.endswith('.txt')]
    num_files = len(filenames)
    print(f'{num_files} .txt files detected in snippets folder...')
    #initialize list to save code snippets
    snippets = list()
    print('Preparing Code Snippets...')
    #preprocess code snippets
    for filename in filenames:

        path = os.path.join(
            directory, 
            filename
            )
        with open(path, 'rb') as f:
            sample = f.readlines()
        
        #decode binary from file and turn it to a string
        sample = [i.decode() for i in sample]
        #replace new-line characters with br
        pattern = re.compile(r'$\n')
        sample = [pattern.sub(' br ', i) for i in sample]
        sample = [re.sub('\n', ' br ', i) if i=='\n' else i for i in sample]
        #join lines to one string
        sample = ' '.join(sample)
        #run the preprocessing on string
        sample = preprocess(sample)
        #append string to code list
        snippets.append(sample)

    #transform the preprocessed code snippets using vectorizer
    snippets = vectorizer.transform(snippets)

    return filenames, snippets, num_files

def load_model():

    """
    Function to load model from pickle file

    Returns
    -------
    model: Scikit-Learn Model Object
        model object constructed from the pickle file
    """

    MODEL_PATH = './model/model.p'
    print('Loading Model...')
    #load architecture from json file
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    return model

def get_predictions(model, filenames, snippets, top_n=1):

    """
    Function to get predictions for the code snippet sequences using the model provided

    Parameters
    ----------
    model: Scikit-Learn Model Object
        Model Object to use for predictions
    filenames: list
        list of strings corresponding to the filenames to be predicted
    snippets: list
        list of preprocessed and padded sequences corresponding to the snippets to be predicted
    top_n: int
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
    pred = model.decision_function(snippets)
    print(f'Making top {top_n} predictions for each snippet')
    for f, p in zip(filenames, pred):
        class_preds = np.argsort(p)[::-1]
        top_n_preds = class_preds[:top_n]
        top_n_preds = [encoder.classes_[i] for i in top_n_preds]
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
        default=1,
        help='The top N predictions to display for each snippet',
        type=int
    )

    return parser.parse_args()

if __name__ == "__main__":

    start = time.time()
    args = _get_args()
    top = args.top
    #get filenames and snippets
    filenames, snippets, num_files = prepare()
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
    seconds = time.time() - start
    print(f'Time summary: {num_files} predictions made in {np.round(seconds, 4)} seconds')

    



