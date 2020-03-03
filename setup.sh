# Setup conda env
conda env create -f environment.yml
conda activate word-homology

# Downliad javaplex
cd plex
wget https://github.com/appliedtopology/javaplex/archive/4.3.4.zip
unzip 4.3.4.zip
rm 4.3.4.zip
cd ..

# Get data files
cd data

wget http://nlp.stanford.edu/data/glove.6B.zip
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json

unzip glove.6B.zip
rm glove.6B.zip glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt
unzip glove.840B.300d.zip
rm glove.840B.300d.zip

wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
gunzip GoogleNews-vectors-negative300.bin.gz
