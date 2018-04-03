#!/bin/zsh

# https://www.clips.uantwerpen.be/conll2000/chunking/
wget https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz
wget https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz
gunzip test.txt.gz
gunzip train.txt.gz

rm train.txt.gz test.txt.gz



# http://www.anc.org/data/masc/downloads/data-download/
#wget http://www.anc.org/MASC/download/MASC-3.0.0.tgz
wget http://www.anc.org/MASC/download/Propbank-original-format.tgz
gunzip Propbank-original-format.tgz


# https://nlp.stanford.edu/data/QuestionBank-Stanford.shtml
#wget https://nlp.stanford.edu/data/QuestionBank-Stanford-1.0.1.zip



# This one is big!
# https://ahclab.naist.jp/resource/tedtreebank/
wget https://ahclab.naist.jp/resource/tedtreebank/naist-ntt-ted-treebank-v1.tar.gz
gunzip naist-tt-ted-treebank-v1.tar.gz



# https://www.kaggle.com/crawford/20-newsgroups/
#wget https://www.kaggle.com/crawford/20-newsgroups/downloads/20-newsgroups.zip/1


# http://www.computing.dcu.ie/~jjudge/qtreebank/4000qs.txt
wget http://www.computing.dcu.ie/~jjudge/qtreebank/4000qs.txt
