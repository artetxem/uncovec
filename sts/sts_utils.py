# Copyright (C) 2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import nltk
import nltk.corpus
import numpy as np
import string


stopwords = set(nltk.corpus.stopwords.words('english'))


def read_embeddings(file, threshold=0, vocabulary=None):
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim)) if vocabulary is None else []
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        if vocabulary is None:
            words.append(word)
            matrix[i] = np.fromstring(vec, sep=' ')
        elif word in vocabulary:
            words.append(word)
            matrix.append(np.fromstring(vec, sep=' '))
    return (words, matrix) if vocabulary is None else (words, np.array(matrix))


def length_normalize_embeddings(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    return matrix / norms[:, np.newaxis]


def read_data(file):
    src, trg, ref = [], [], []
    for line in file:
        cols = line.split('\t')
        src.append(cols[5])
        trg.append(cols[6])
        ref.append(float(cols[4]))
    return src, trg, np.array(ref)


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def recase(sent, word2ind):
    ans = []
    for word in sent:
        if word in word2ind and word.lower() in word2ind:
            if word2ind[word.lower()] < word2ind[word]:
                ans.append(word.lower())
            else:
                ans.append(word)
        elif word.lower() in word2ind:
            ans.append(word.lower())
        else:
            ans.append(word)
    return ans


def strip_punctuation(sentence):
    return [word for word in sentence if not all([c in string.punctuation for c in word])]


def remove_stopwords(sentence):
    return [word for word in sentence if word not in stopwords]


def remove_oovs(sentence, vocab):
    return [word for word in sentence if word in vocab]


def pearson(sys, ref):
    return np.corrcoef(sys, ref)[0, 1]


def cosine(a, b):
    return a.dot(b) / np.sqrt(a.dot(a)*b.dot(b))


def centroid(sent, emb, word2ind):
    return sum([emb[word2ind[word]] for word in sent]) / len(sent)


def centroid_cosine(src, trg, emb, word2ind, backoff=1.0):
    src = remove_oovs(src, word2ind)
    trg = remove_oovs(trg, word2ind)
    if len(src) == 0 or len(trg) == 0:
        return backoff
    else:
        return cosine(centroid(src, emb, word2ind), centroid(trg, emb, word2ind))
