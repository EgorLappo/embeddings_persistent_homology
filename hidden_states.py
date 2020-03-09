import numpy as np
import string
import re
from os import path

from allennlp.commands.elmo import ElmoEmbedder

import torch
from transformers import BertTokenizer, BertModel, BertConfig

import multiprocessing

import persim
from ripser import ripser
import matplotlib
from matplotlib import pyplot as plt

import argparse


parser = argparse.ArgumentParser(description="embed the text as a point cloud in the embedding space")
parser.add_argument('text', help='input text file')
parser.add_argument('-e', '--embed', action="store_true", help="save embedding vectors in numpy format")
parser.add_argument('-r', '--run', action="store_true", help="run ripser computation")
parser.add_argument('-d', '--dim', type=int, default=2, help="max dimension to run")
parser.add_argument('-n', '--sample', type=int, default=100, help="points to subsample")
args = parser.parse_args()


def tokenize_from_file(filename, keep_punct = False):
    with open(filename,"r") as file:
        text = file.read()
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


def ripser_compute(points):
    return ripser(points, maxdim=args.dim, n_perm=args.sample)['dgms']

if __name__ == "__main__":
    text_name = args.text
    text_em = "embeddings/" + args.text
    text_results = "ripser_output/hidden_states/" + args.text
    text_filename = "texts/" + args.text
    text_words = tokenize_from_file(text_filename)

    print("read text of length " + str(len(text_words)) + " words...")

    if args.embed:

        #BERT
        print("embedding with BERT...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        with open("texts/poem_1", "r") as f:
            text = f.read()
        tokenized_text = [tokenizer.tokenize('[CLS] ' + x + ' [SEP]') for x in text.split('.')]
        indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
        tokens_tensors = [torch.tensor([x]) for x in indexed_tokens]
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        model.eval()

        with torch.no_grad():
            outputs = [model(x) for x in tokens_tensors]
            bert_embeddings = torch.stack([x for output in outputs for x in output[0][0][1:-1]])

        hidden_states_bert = []

        for i in range(13):
            hidden_state_bert_i = torch.stack([x for output in outputs for x in output[2][i][0][1:-1]])
            hidden_states_bert.append(hidden_state_bert_i)
            np.save(text_em + "hidden_states_"+str(i)+".bert", hidden_state_bert_i.numpy())   

        #ELMo
        print("embedding with ELMo")
        elmo_embedder = ElmoEmbedder(options_file="data/elmo_2x2048_256_2048cnn_1xhighway_options.json", \
                                weight_file="data/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5")

        hidden_states_elmo = elmo_embedder.embed_sentence(text_words)


        for i in range(3):
            np.save(text_em + "hidden_states_"+str(i)+".elmo", hidden_states_elmo[i])

    hidden_states_bert = []
    for i in range(13):
        state = np.load(text_em + "hidden_states_"+str(i)+".bert.npy")
        hidden_states_bert.append(state)

    hidden_states_elmo = []
    for i in range(3):
        state = np.load(text_em + "hidden_states_"+str(i)+".elmo.npy")
        hidden_states_elmo.append(states)

    if args.run:

        print("launching ripser in parallel")
        pool = multiprocessing.Pool()
        diagrams = pool.map(ripser_compute, [hidden_states_bert[0], hidden_states_bert[3], \
                                             hidden_states_bert[6], hidden_states_bert[9], hidden_states_bert[12], \
                                             hidden_states_elmo[0], hidden_states_elmo[1], hidden_states_elmo[2]])
        pool.close()

        diagrams = list(diagrams)

        print("making plots")
        plt.figure(figsize=(25,10))
        plt.subplot(161)
        persim.plot_diagrams(diagrams[0])
        plt.title("BERT, hidden state $1$")

        plt.subplot(162)
        persim.plot_diagrams(diagrams[1])
        plt.title("BERT, hidden state $4$")

        plt.subplot(163)
        persim.plot_diagrams(diagrams[2])
        plt.title("BERT, hidden state $7$")

        plt.subplot(164)
        persim.plot_diagrams(diagrams[3])
        plt.title("BERT, hidden state $10$")

        plt.subplot(165)
        persim.plot_diagrams(diagrams[4])
        plt.title("BERT, hidden state $13$")


        plt.savefig(text_results + "bert-hidden-plots.pdf", dpi=500)

        plt.clf()

        plt.figure(figsize=(15,8))
        plt.subplot(131)
        persim.plot_diagrams(diagrams[5])
        plt.title("ELMo, hidden state $1$")

        plt.subplot(132)
        persim.plot_diagrams(diagrams[6])
        plt.title("ELMo, hidden state $2$")

        plt.subplot(133)
        persim.plot_diagrams(diagrams[7])
        plt.title("ELMo, hidden state $3$")

        plt.savefig(text_results + "elmo-hidden-plots.pdf", dpi=500)


        # print("writing output")
        # for i, emb in enumerate(["word2vec", "glove_wiki", "glove_cc", "elmo", "bert"]):
        #     with open(text_results + "." + emb,"w") as f:
        #         for dim in diagrams[i]:
        #             for interval in dim:
        #                 f.write(str(interval[0])+" "+str(interval[1])+"\n")
        #             f.write("\n")

