import string
import re
from os import path

import torch
from transformers import BertTokenizer, BertModel

import multiprocessing

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
from tqdm import trange

import argparse


parser = argparse.ArgumentParser(description="make silhouette plots")
parser.add_argument('text', help='input text file')
parser.add_argument('-k', type=int, default=100, help='max k for k-means')
args = parser.parse_args()

def silhouette_compute(arg):
    k = args.k
    i, points = arg
    return [silhouette_score(points, KMeans(n_clusters=i).fit_predict(points)) for i in trange(2, k, position=i)]

def make_plot(y):
    plt.bar(x=np.arange(2, 2 + len(y)), height=np.array(y), width=1.)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")


if __name__ == "__main__":
    text_name = args.text
    text_em = "embeddings/" + args.text
    text_results = "ripser_output/" + args.text
    text_filename = "texts/" + args.text

    # load embeddings
    word2vec_embeddings = np.load(text_em + ".word2vec.npy")
    glove_wiki_embeddings = np.load(text_em + ".glove_wiki.npy")
    glove_cc_embeddings = np.load(text_em + ".glove_cc.npy")
    elmo_embeddings = np.load(text_em + ".elmo.npy")
    bert_embeddings = np.load(text_em + ".bert.npy")

    print("running silhouette computation")
    pool = multiprocessing.Pool()
    diagrams = pool.map(silhouette_compute, enumerate([word2vec_embeddings, glove_wiki_embeddings, 
                        glove_cc_embeddings, elmo_embeddings, bert_embeddings]))
    pool.close()

    print("making plots")
    plt.figure(figsize=(25,8))
    plt.subplot(151)
    make_plot(diagrams[0])
    plt.title("word2vec")

    plt.subplot(152)
    make_plot(diagrams[1])
    plt.title("GLoVe Wiki")

    plt.subplot(153)
    make_plot(diagrams[2])
    plt.title("GloVe CC")

    plt.subplot(154)
    make_plot(diagrams[3])
    plt.title("ELMo")

    plt.subplot(155)
    make_plot(diagrams[4])
    plt.title("BERT")

    plt.savefig(text_results + "-silhouette.pdf", dpi=500)


