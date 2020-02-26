sudo apt-get install git-lfs
git lfs install
git clone https://github.com/mmihaltz/word2vec-GoogleNews-vectors.git
mv word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin.gz .
rm -rf word2vec-GoogleNews-vectors

wget http://nlp.stanford.edu/data/glove.6B.zip
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json

unzip glove.6B.zip
rm glove.6B.zip glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
gunzip GoogleNews-vectors-negative300.bin.gz

pip install numpy gensim torch allennlp transformers

wget https://github.com/appliedtopology/javaplex/archive/4.3.4.zip
unzip 4.3.4.zip
rm 4.3.4.zip