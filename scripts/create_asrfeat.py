#!/bin/python
import numpy
import os
import pickle
import sys
import nltk
import collections
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import LancasterStemmer
from nltk import sent_tokenize, word_tokenize, pos_tag
from gensim.models import Word2Vec
import warnings
from scipy import sparse

warnings.filterwarnings(action = 'ignore')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} vocab_file, file_list".format(sys.argv[0]))
        print("vocab_file -- path to the vocabulary file")
        print("file_list -- the list of videos")
        exit(1)

    tag_map = defaultdict(lambda: None)
    tag_map['N'] = wn.NOUN
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    # lemmatizer = WordNetLemmatizer()

    asr_dir = sys.argv[1]
    feat_dim = int(sys.argv[2])
    output_dir = sys.argv[3]

    asr_list = [f.split('.')[0] for f in os.listdir(asr_dir) if f.split('.')[1] == 'txt']
    asr_tokens = []
    punct = "?:!.,;"
    lancaster = LancasterStemmer()
    for idx, asr in enumerate(asr_list):
        cur_tokens = []
        if idx % 100 == 0:
            print('{}th file processing'.format(idx))

        with open(os.path.join(asr_dir, '{}.txt'.format(asr)), 'r') as f:
            sentence = ' '.join([line for line in f.readlines()])
            tokens = word_tokenize(sentence)
            for t in tokens:
                if t in punct:
                    continue
                cur_tokens.append(lancaster.stem(t))
        asr_tokens.append(cur_tokens)

    word_vec_model = Word2Vec(asr_tokens, size=feat_dim, min_count=1, window=5, iter=20, sg=1)
    words = list(word_vec_model.wv.vocab)
    word_idx = {}
    for idx, word in enumerate(words):
        word_idx[word] = idx

    video_idx = {}
    for idx, video in enumerate(asr_list):
        video_idx[video] = idx

    row = []
    col = []
    data = []
    for idx, cur_token in enumerate(asr_tokens):
        counter = collections.Counter(cur_token)
        for w in counter:
            row.append(idx)
            col.append(word_idx[w])
            data.append(numpy.log(counter[w] + 1))

    asr_word_mtx = sparse.csr_matrix((numpy.array(data), (numpy.array(row), numpy.array(col))))
    idf_vec = numpy.log(len(asr_list) / (asr_word_mtx > 0).astype(int).sum(axis=0))
    idf_weighted_mtx = numpy.matmul(numpy.diag(idf_vec.A1), word_vec_model.wv.syn0)
    asr_video_mtx = asr_word_mtx * idf_weighted_mtx        

    video_list = [line.strip() for line in open('list/all.video', 'r')]
    video_feat = numpy.zeros((len(video_list), asr_video_mtx.shape[1]))
    for idx, video in enumerate(video_list):
        if video in asr_list:
            video_feat[idx] = asr_video_mtx[video_idx[video]]

    numpy.savetxt(os.path.join(output_dir, 'result.csv'), video_feat, delimiter=',')

    print('ASR features generated successfully!')
