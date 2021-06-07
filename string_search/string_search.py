import pandas as pd
import string
import os
import numpy as np
from scipy.sparse import csr_matrix

BIN_EDGES = [0.8, 0.6, 0.4]
BINS = [0.0, 0.4, 0.6, 0.8, 1.0] # Used by sparse
N_GRAM_SIZE = 3

CHAR_MAPPING = {
    'ş': 's',
    'ı': 'i',
    'ö': 'o',
    'ğ': 'g',
    'ç': 'c',
    'ü': 'u'
}
CHAR_MAPPING = {ord(k): ord(v) for k, v in CHAR_MAPPING.items()}

def get_n_grams(s):
    n_grams = []
    for i in range(len(s) - N_GRAM_SIZE + 1):
        ngram_str = s[i: i + N_GRAM_SIZE]
        #if ' ' not in ngram_str[1:]:
        if ngram_str[1] != ' ':
            n_grams.append(ngram_str)
        
    
    return reduce_n_grams(n_grams)

def reduce_n_grams(n_grams):
    n_grams_single = []
    for token in n_grams:
        if token not in n_grams_single:
            n_grams_single.append(token)
    return n_grams_single

def preprocess(s: str):
    s = s.strip().replace('İ', 'i').lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.translate(CHAR_MAPPING)
    return s



def create_ngram_index(n_grams, n_index_tokens = 1000):
    """
    Args:
        n_grams: a list that contains a list of n-grams for each entry.
    """
    n_gram_index = {}

    for i, tokens in enumerate(n_grams):
        for token in tokens[:n_index_tokens]:
            # Create new entry
            if token not in n_gram_index:
                n_gram_index[token] = [i]

            # Add index to existing token
            else:
                n_gram_index[token].append(i)
    return n_gram_index

def sort_matches(matches):
    match_vals = list(matches.values())
    match_keys = list(matches.keys())

    match_idx = list(reversed(np.argsort(match_vals)))
    match_vals = [match_vals[i] for i in match_idx]
    match_keys = [match_keys[i] for i in match_idx]

    return match_keys, match_vals

def get_matches(input_n_grams, n_gram_index, n_gram_lengths):
    input_n_gram_set = input_n_grams
    input_len = len(input_n_gram_set)
    matches = {}

    for token in input_n_gram_set:
        if token in n_gram_index:
            token_matches = n_gram_index[token]

            for token_match in token_matches:
                if token_match in matches:
                    # Increase matching token count of this entry
                    matches[token_match] += 1
                else:
                    # Add new entry
                    matches[token_match] = 1

    #matches = {k: v / max(input_len, n_gram_lengths[k]) for k,v in matches.items()}
    matches = {k: v / n_gram_lengths[k] for k,v in matches.items()}
    #matches = {k: v / input_len for k,v in matches.items()}

    return [(k, v) for k, v in matches.items()]



def sort_bins(match_kvs):
    return sorted(match_kvs, key = lambda x: x[1], reverse = True)

def bin_matches(matches, get_sorted = True):
    exact_matches = []
    match_bins = {e: [] for e in BIN_EDGES + [0.0]}

    for match in matches:
        k, v = match
        if v == 1.0:
            # Get exact matches
            exact_matches.append(match)
        else:
            # Place matches into bins
            has_bin = False
            for e in BIN_EDGES:
                if v >= e:
                    match_bins[e].append(match)
                    has_bin = True
                    break

            if not has_bin:
                # Mathces smaller than the smallest edge go into 0.0 bin
                match_bins[0.0].append(match)

    match_bins[1.0] = exact_matches

    # Sort each bin
    if get_sorted:
        match_bins = {k: sort_bins(v) for k, v in match_bins.items()}

    return match_bins

def get_match_values(match_bins, entries):
    return {k: [(v[0], entries[v[0]], v[1]) for v in vals] for k, vals in match_bins.items()}

def search_groups_vectorized(input_n_grams, n_gram_index, vocabulary, n_gram_lengths, get_sorted = False, search_person = True):
    input_n_gram_set = input_n_grams#reduce_n_grams(input_n_grams)
    input_len = len(input_n_gram_set)

    token_to_records = [vocabulary.get(token) for token in input_n_gram_set if token in vocabulary]
    n_gram_sum = np.squeeze(np.asarray(n_gram_index[:, token_to_records].sum(axis = 1)))

    # Normalize similarities by length
    if search_person:
        n_gram_sum = n_gram_sum / len(input_n_gram_set)#np.maximum(n_gram_lengths, len(input_n_gram_set))
    else:
        n_gram_sum = n_gram_sum / len(input_n_gram_set)

    matches = n_gram_sum
    match_bins = np.digitize(matches, BINS)

    match_data = pd.DataFrame({'bin': match_bins, 'value': matches})
    match_data['i'] = match_data.index
    match_groups = dict(tuple(match_data.groupby('bin')))

    match_groups = {BINS[k-1]: list(zip(v.i.values, v.value.values)) for k, v in match_groups.items()}
    for b in BINS:
        if b not in match_groups:
            match_groups[b] = []

    # Sort each bin
    if get_sorted:
        match_groups = {k: sort_bins(v) for k, v in match_groups.items()}
    return match_groups

def create_ngram_index_sparse(n_grams, first_n_tokens = 1000):
    """
    Args:
        n_grams: a list that contains a list of n-grams for each entry.
    """
    indptr = [0]
    indices = []
    data = []
    vocabulary = {} # token idx
    lengths = [] # records lengths

    for entry in n_grams:
        entry_reduced = entry#reduce_n_grams(entry) # Remove duplicate n-grams
        lengths.append(min(max(len(entry_reduced), 1), first_n_tokens))

        for token in entry[:first_n_tokens]:
            index = vocabulary.setdefault(token, len(vocabulary)) # index of the record with this token
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))

    return csr_matrix((data, indices, indptr), dtype = 'int8'), vocabulary, lengths