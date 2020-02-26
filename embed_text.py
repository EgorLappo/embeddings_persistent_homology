import numpy as np
import string
import re

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

from allennlp.commands.elmo import ElmoEmbedder

import torch
from transformers import BertTokenizer, BertModel

import argparse
parser = argparse.ArgumentParser(description="embed the text as a point cloud in the embedding space")
parser.add_argument('text', help='input text file')
parser.add_argument('-s', '--save_numpy', action="store_true", "save vectors in numpy format")
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

def vectors_to_plex(vectors, filename):
    #dist_matrix = euclidean_distances(vectors)
    #vectors = vectors / np.mean(dist_matrix)
    with open(filename, "w") as out:
        #out.write(" ".join([str(i) for i in range(vectors.shape[1])]))
        #out.write("\n")
        for i, vector in enumerate(vectors):
            out.write(" ".join(map(str,vector)))
            out.write("\n")

if __name__ == "__main__":
    text_filename = args.text
    text_words = tokenize_from_file(text_filename)

    #word2vec
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', \
                                                                     binary=True)  
    word2vec_embeddings = get_vectors(word2vec_model, text_words)
    if args.save_numpy:
        np.save(text_filename+".word2vec", word2vec_embeddings)                                                           

    #GLoVe
    glove2word2vec(glove_input_file="glove.6B.300d.txt", word2vec_output_file="glove.6B.300d.gensim.txt")
    glove2word2vec(glove_input_file="glove.840B.300d.txt", word2vec_output_file="glove.840B.300d.gensim.txt")
    glove_cc_model = gensim.models.KeyedVectors.load_word2vec_format('glove.840B.300d.gensim.txt', binary=False)
    glove_wiki_model = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.300d.gensim.txt', binary=False)

    glove_cc_embeddings = get_vectors(glove_cc_model, text_words)
    glove_wiki_embeddings = get_vectors(glove_wiki_model, text_words)
    
    if args.save_numpy:
        np.save(text_filename+".glove_cc", glove_cc_embeddings)
        np.save(text_filename+".glove_wiki", glove_wiki_embeddings)

    #ELMo
    elmo_embedder = ElmoEmbedder(options_file = "elmo_2x2048_256_2048cnn_1xhighway_options.json", \
                            weight_file = "elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5")

    elmo_embeddings_raw = elmo_embedder.embed_sentence(text_words)

    elmo_embeddings = elmo_embeddings_raw[2]

    if args.save_numpy:
        np.save(text_filename+".elmo",elmo_embeddings)

    #BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open(text_filename, "r") as f:
        text = f.read()
    tokenized_text = tokenizer.tokenize(text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])

    model = BertModel.from_pretrained('bert-base-uncased')

    model.eval()

    with torch.no_grad():
        outputs = model(tokens_tensor)
        
        bert_embeddings = outputs[0][0]
    
    if args.save_numpy:    
        np.save(text_filename+".bert", bert_embeddings.numpy())   

    # prepare for Plex
    vectors_to_plex(word2vec_embeddings, filename=text_filename+".word2vec_plex")
    vectors_to_plex(glove_cc_embeddings, filename=text_filename+".glove_cc_plex")
    vectors_to_plex(glove_wiki_embeddings, filename=text_filename+".glove_wiki_plex")
    vectors_to_plex(elmo_embeddings, filename=text_filename+".elmo_plex")
    vectors_to_plex(bert_embeddings, filename=text_filename+".bert_plex")
