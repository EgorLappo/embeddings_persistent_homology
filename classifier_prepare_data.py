import numpy as np
import string
import re
import os
from os import path
import random

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import *

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

from allennlp.commands.elmo import ElmoEmbedder

import torch
from transformers import BertTokenizer, BertModel

import multiprocessing

import persim
from ripser import ripser

import matplotlib
from matplotlib import pyplot as plt


import tqdm

import pandas as pd

import argparse



parser = argparse.ArgumentParser(description="construct training and testing data")
parser.add_argument('-n', '--sample', type=int, default=200, help="points to subsample")
parser.add_argument('-s', '--split', type=list, default=[0.6, 0.2, 0.2], help="training-validation-testing split")
parser.add_argument('-i', '--min-len', type=list, default=300, help="minimum text length")
parser.add_argument('-a', '--max-len', type=list, default=1200, help="maximum text length")
parser.add_argument('-k', type=int, default=100, help='max k for k-means')
parser.add_argument('-l', type=int, default=5, help='silhouette k step size')
parser.add_argument('-t', type=int, default=100, help='time discretization')
args = parser.parse_args()

assert sum(args.split) == 1, 'invalid data split'


def silhouette_compute(points):
    return np.array([silhouette_score(points, KMeans(n_clusters=i, max_iter=100).fit_predict(points)) 
                        for i in range(2, args.k, args.l)])

def tokenize(text, keep_punct = False):
    if keep_punct is True:
        for punct in string.punctuation:
            text = text.replace(punct, ' ' + punct + ' ')
    else:
        for punct in string.punctuation:
            text = text.replace(punct, ' ')
    
    text = re.sub('\s+', ' ', text)
    
    result = []
    
    for x in text.lower().split(' '):
        if x.isalpha():
            result.append(x)
        else:
            word = []
            for y in x: # for every character
                if y.isalpha(): word.append(y)
            if len(word) > 0:
                result.append(''.join(word))
                
    return result

def get_vectors(wv, words):
    M = []
    for w in words:
        try:
            M.append(wv[w])
        except KeyError:
            continue
    M = np.stack(M)
    return M

TIME_DISC = args.t

def ripser_compute(points):
    diagrams = ripser(points, maxdim=2, n_perm=args.sample)['dgms']
    norm = pdist(points).max()
    for intervals in [diagrams[1], diagrams[2]]:
        arr = np.zeros(TIME_DISC)
        for start, end in intervals:
            for i in range(int(start / norm * TIME_DISC), int(end / norm * TIME_DISC)):
                arr[i] += 1
        yield arr

def init_worker():
    global elmo_embedder
    cuda_id = int(multiprocessing.current_process().name.split('-')[1]) - 1
    print('Worker on CUDA device', cuda_id)
    elmo_embedder = ElmoEmbedder(options_file="data/elmo_2x2048_256_2048cnn_1xhighway_options.json", \
                        weight_file="data/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5", cuda_device=random.randrange(4))

def embed(text_words):
    text_tok = [tokenize(x) for x in text_words.split('.')]
    text_tok = [x for x in text_tok if x]

    if not (args.min_len <= sum(len(x) for x in text_tok) < args.max_len):
        return None

    elmo_embeddings_raw = []
    while text_tok:
        minibatch = 10
        batch, text_tok = text_tok[:minibatch], text_tok[minibatch:]
        elmo_embeddings_raw.extend(elmo_embedder.embed_batch(batch))

    elmo_embeddings = np.concatenate([x[-1] for x in elmo_embeddings_raw], axis=0)

    return elmo_embeddings

def embed_news(x):
    return embed(x), 1

def embed_poem(x):
    return embed(x), 0

def compute_results(xy):
    x, y = xy
    h1, h2 = ripser_compute(x)
    sil = silhouette_compute(x)
    return (sil, h1, h2), y


if __name__ == "__main__":
    pool = multiprocessing.Pool(4, initializer=init_worker)

    if path.exists('embeddings/cache.npy'):
        embedded = np.load('embeddings/cache.npy', allow_pickle=True)
    else:
        embedded = []
        news_files = [open(path.join(root, f), 'rb').read().decode('utf-8', 'ignore').strip() 
                        for root, _, file_list in os.walk('train_data/20news-18828') 
                        for f in file_list]
        embedded.extend(tqdm.tqdm(pool.imap(embed_news, news_files), desc='embed news', total=len(news_files)))
        df = pd.read_csv('train_data/kaggle_poem_dataset.csv')
        embedded.extend(tqdm.tqdm(pool.imap(embed_poem, df.Content.values), desc='embed poems', total=len(df.Content.values)))
        embedded = [x for x in embedded if x[0] is not None]
        random.shuffle(embedded)
        np.save('embeddings/cache.npy', embedded)

    print(f'embedded {sum(y == 0 for x, y in embedded)} news articles') 
    print(f'embedded {sum(y == 1 for x, y in embedded)} poems') 

    silhouette_results = []
    ripser_h1_results = []
    ripser_h2_results = []

    for itr, (result, y) in enumerate(tqdm.tqdm(
                                pool.imap(compute_results, embedded), 
                                desc='computing results', total=len(embedded))):
        sil, h1, h2 = result
        silhouette_results.append((sil, y))
        ripser_h1_results.append((h1, y))
        ripser_h2_results.append((h2, y))
        if itr % 10: continue
        for location, data in [('silhouette_data', silhouette_results),
                               ('ripser_data_h1', ripser_h1_results),
                               ('ripser_data_h2', ripser_h2_results)]:
            scaled_splits = [int(len(data) * s) for s in args.split]
            cut = [sum(scaled_splits[:i]) for i in range(4)]
            np.save(f'{location}/train.npy', data[cut[0]:cut[1]])
            np.save(f'{location}/val.npy', data[cut[1]:cut[2]])
            np.save(f'{location}/test.npy', data[cut[2]:cut[3]])

    pool.close()

