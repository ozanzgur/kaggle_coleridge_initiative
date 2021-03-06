import os
import logging
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import re
#from nltk.tokenize import wordpunct_tokenize
#logger = logging.getLogger('pipeline')

def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())

digits = '0123456789'
dataset_names = list(pd.read_csv('data/kaggle_22k_datasets.csv').title.values) +\
                list(pd.read_csv('data/kaggle_800_datasets.csv').title.values)
dataset_names = [clean_text(l).lower() for l in dataset_names]
dataset_names = [l.split(' ') for l in dataset_names]

dataset_name_token_count = 8
dataset_tokens = [[] for _ in range(dataset_name_token_count)]

for i in range(dataset_name_token_count):
    new_tokens_index = []
    for name_tokens in dataset_names:
        if i < len(name_tokens):
            new_token = name_tokens[i]
            if len(new_token) > 2:
                new_tokens_index.append(new_token)
        else:
            break

dataset_tokens_after_first = []
for tokens in dataset_tokens[1:]:
    dataset_tokens_after_first.extend(tokens)

dataset_tokens = [dataset_tokens[0], dataset_tokens_after_first]
dataset_tokens = [set(t) for t in dataset_tokens]

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

for t in ['of', 'the', 'and']:
    stop_words.remove(t)

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")
digits = '0123456789'

def get_features(s):
    return [(w, t) for w, t in zip(\
            s['TOKEN'].values.tolist(),
            s['TARGET'].values.tolist()                          
        )]

import string
puncs = [c for c in string.punctuation]

def mask_numbers(text):
        # Replace each numeric char with '#'
        
        def repl(m):
            return f" {'#' * len(m.group())} "
        text = re.sub(r'[0-9]+', repl, text)
        return text

def make_single_whitespace(text):
    return _RE_COMBINE_WHITESPACE.sub(" ", text).strip()

class TextFeatureExtractor:
    def __init__(self, *args, **kwargs):
        # Initialize super
        
        # Load parameters
        def_args = dict()
        
        # Extract related arguments
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})

    def transform(self, x):
        x.update({'output': [_df2features(x['output'])]})
        return x

    def _mask_numbers(self, text):
        # Replace each numeric char with #
        
        def repl(m):
            return f" {'#' * len(m.group())} "
        text = re.sub(r'[0-9]+', repl, text)
        return text

    def fit_transform(self, data, train_filenames, val_filenames):
        self.train_filenames = train_filenames
        self.val_filenames = val_filenames

        
        output = {}

        # Process each set
        for setname in ['train', 'val']:
            docs = []
            for f in tqdm(self.__dict__.get(f'{setname}_filenames')):
                df_slice = data[f]

                assert not df_slice['TOKEN'].isnull().any(), 'All tokens must have a value'
                df_slice['TARGET'] = df_slice['TARGET'].fillna('OTHER')
                df_slice['TOKEN'] = df_slice['TOKEN'].values.astype('U')
                df_slice['TOKEN'] = df_slice['TOKEN'].apply(mask_numbers)
                df_slice['TOKEN'] = df_slice['TOKEN'].apply(make_single_whitespace)

                data_slice = get_features(df_slice)
                docs.append(data_slice)
            
            X = [_doc2features(s) for s in tqdm(docs)]
            y = [_doc2labels(s) for s in tqdm(docs)]

            del docs
            
            assert(len(X) == len(y))
            
            output[f'{setname}_data'] = (X, y)
        
        return output

class DocGetter(object):
    def __init__(self, data):
        self.n_doc = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(\
            s['TOKEN'].values.tolist(), 
            s['TARGET'].values.tolist()
                                                                     
        )]
        self.grouped = self.data.groupby('FILENAME').apply(agg_func)
        self.docs = [s for s in self.grouped]

"""def _text2features(text):
    #words = wordpunct_tokenize(text)
    return [_word2features(words, i) for i in range(len(words))]"""

def _df2features(df):
    """Returns a list of examples.
    """
    for col in ['TOKEN']:
        assert(col in df.columns)
    
    df['TOKEN'] = df['TOKEN'].values.astype('U')
    df['TOKEN'] = df['TOKEN'].apply(mask_numbers)
    df['TOKEN'] = df['TOKEN'].apply(make_single_whitespace)
    
    agg_func = lambda s: [(w,) for w in s['TOKEN'].values.tolist()]
    feature_list = agg_func(df)
    words = [s for s in feature_list]
    
    return _doc2features(words)

def _word2features(words, i):
    word = words[i]
    if not isinstance(word, str):
        word = word[0]
    
    # isfirstname = fe_isname.is_firstname(word)
    # islastname = fe_isname.is_lastname(word)
    # isname = isfirstname or islastname
    
    # if isfirstname and islastname:
    #     word = '#NAME#'
    # elif isfirstname:
    #     word = '#FIRSTNAME#'
    # elif islastname:
    #     word = '#LASTNAME#'
    word_lower = word.lower()

    digit_count = len([c for c in word if c in digits])
    length = len(word)
    assert length > 0, "All tokens must have length > 0"

    features = {
        #'bias': 1.0,
        #'word.index': i
    }
    
    # Add line & word indices
    # features.update({
    #     'word.lineindex': li,
    #     'word.wordindex': wi,
        
    # })
    
    # if isname:
    #     features.update({
    #     'word.lower()': word
        
    # })
    #else:
    # If all digits
    if word.isdigit():
        features.update({
            'd': True,
            'dc': digit_count,
            '4d': digit_count == 4,
            'dg': 1.0,
            'l': length
        })
    else: # Not all digit
        features.update({
            'd': False,
            'dg': digit_count / length,
            'dc': digit_count,
            'l': length,
            #'w': word.lower(),
            'u': word[0].isupper(),
            'Au': word.isupper(),
            's': word in stop_words,
            'uc': len([c for c in word if c.isupper()]),
            'daf': word_lower in dataset_tokens[0],
            'da': word_lower in dataset_tokens[1]
        })

    if i > 0:
        word_other = words[i-1][0]
        features.update({
            #'-1': word_other.lower(),
            '-1u': word_other[0].isupper(),
            '-1Au': word_other.isupper(),
            '-1s': word_other in stop_words,
            #'-1w': word_other.lower(),
            '-1uc': len([c for c in word_other if c.isupper()]),
            '-1daf': word_other.lower() in dataset_tokens[0],
            '-1da': word_other.lower() in dataset_tokens[1]
        })
        if i > 1:
            word_other = words[i-2][0]
            features.update({
                '-2u': word_other[0].isupper(),
                '-2Au': word_other.isupper(),
                '-2s': word_other in stop_words,
                '-2daf': word_other.lower() in dataset_tokens[0],
                '-2da': word_other.lower() in dataset_tokens[1],
                #'-2w': word_other.lower(),
            })
            if i > 2:
                word_other = words[i-3][0]
                features.update({
                    '-3u': word_other[0].isupper(),
                    '-3Au': word_other.isupper(),
                    '-3s': word_other in stop_words
                })
                if i > 3:
                    word_other = words[i-4][0]
                    features.update({
                        '-4u': word_other[0].isupper(),
                        '-4Au': word_other.isupper(),
                        '-4s': word_other in stop_words
                    })

    if i < len(words)-1:
        word_other = words[i+1][0]
        features.update({
            #'+1':  word_other.lower(),
            '+1u': word_other[0].isupper(),
            '+1Au': word_other.isupper(),
            '+1s': word_other in stop_words,
            '+1uc': len([c for c in word_other if c.isupper()]),
            '+1daf': word_other.lower() in dataset_tokens[0],
            '+1da': word_other.lower() in dataset_tokens[1],
            #'+1w': word_other.lower()
        })
        if i < len(words)-2:
            word_other = words[i+2][0]
            features.update({
                #'+2':  word_other.lower(),
                '+2u': word_other[0].isupper(),
                '+2Au': word_other.isupper(),
                '+2s': word_other in stop_words,
                '+2uc': len([c for c in word_other if c.isupper()]),
                '+2daf': word_other.lower() in dataset_tokens[0],
                '+2da': word_other.lower() in dataset_tokens[1],
                #'+2w': word_other.lower()
            })
            if i < len(words)-3:
                word_other = words[i+3][0]
                features.update({
                    #'+3':  word_other.lower(),
                    '+3u': word_other[0].isupper(),
                    '+3Au': word_other.isupper(),
                    '+3s': word_other in stop_words,
                    '+3uc': len([c for c in word_other if c.isupper()]),
                    '+3daf': word_other.lower() in dataset_tokens[0],
                    '+3da': word_other.lower() in dataset_tokens[1],
                    #'+3w': word_other.lower()
                })
                if i < len(words)-4:
                    word_other = words[i+4][0]
                    features.update({
                        #'+4':  word_other.lower(),
                        '+4u': word_other[0].isupper(),
                        '+4Au': word_other.isupper(),
                        '+4s': word_other in stop_words,
                        '+4uc': len([c for c in word_other if c.isupper()]),
                        '+4daf': word_other.lower() in dataset_tokens[0],
                        '+4da': word_other.lower() in dataset_tokens[1],
                        #'+4w': word_other.lower()
                    })
                    if i < len(words)-5:
                        word_other = words[i+5][0]
                        features.update({
                            #'+5':  word_other.lower(),
                            '+5u': word_other[0].isupper(),
                            '+5Au': word_other.isupper(),
                            '+5s': word_other in stop_words,
                            '+5uc': len([c for c in word_other if c.isupper()]),
                            '+5daf': word_other.lower() in dataset_tokens[0],
                            '+5da': word_other.lower() in dataset_tokens[1],
                            #'+5w': word_other.lower()
                        })
                        if i < len(words)-6:
                            word_other = words[i+6][0]
                            features.update({
                                #'+6':  word_other.lower(),
                                '+6u': word_other[0].isupper(),
                                '+6Au': word_other.isupper(),
                                '+6s': word_other in stop_words,
                                '+6uc': len([c for c in word_other if c.isupper()]),
                                '+6daf': word_other.lower() in dataset_tokens[0],
                                '+6da': word_other.lower() in dataset_tokens[1],
                                #'+6w': word_other.lower()
                            })
        
    return features

def _doc2features(doc):
    """Returns a list of examples.
    """
    words = [(ex[0],) for ex in doc]
    return [_word2features(words, i) for i in range(len(words))]

def _doc2labels(doc):
    return [s[-1] for s in doc]
def _doc2tokens(doc):
    return [s[0] for s in doc]