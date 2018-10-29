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

import sts_utils as utils

import argparse
import numpy as np


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a simple STS baseline')
    parser.add_argument('data', help='test file')
    parser.add_argument('embeddings', help='the word embeddings')
    parser.add_argument('--normalize', action='store_true', help='length normalize word embeddings')
    parser.add_argument('--keep_stopwords', action='store_true', help='do not remove stopwords')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input (defaults to utf-8)')
    args = parser.parse_args()

    # Read embeddings
    f = open(args.embeddings, encoding=args.encoding, errors='surrogateescape')
    words, emb = utils.read_embeddings(f)

    # Build word to index map
    word2ind = {word: i for i, word in enumerate(words)}

    # Length normalize embeddings
    if args.normalize:
        emb = utils.length_normalize_embeddings(emb)

    # Read data
    src, trg, ref = utils.read_data(open(args.data, encoding=args.encoding, errors='surrogateescape'))

    # Tokenize
    src = [utils.tokenize(sent) for sent in src]
    trg = [utils.tokenize(sent) for sent in trg]

    # Recase
    src = [utils.recase(sent, word2ind) for sent in src]
    trg = [utils.recase(sent, word2ind) for sent in trg]

    # Strip punctuation
    src = [utils.strip_punctuation(sent) for sent in src]
    trg = [utils.strip_punctuation(sent) for sent in trg]

    # Remove stopwords
    if not args.keep_stopwords:
        src = [utils.remove_stopwords(sent) for sent in src]
        trg = [utils.remove_stopwords(sent) for sent in trg]

    # Compute similarities
    sys = np.zeros(ref.shape)
    for i in range(ref.shape[0]):
        sys[i] = utils.centroid_cosine(src[i], trg[i], emb, word2ind)

    # Compute score
    print('{:.4f}'.format(utils.pearson(sys, ref)))


if __name__ == '__main__':
    main()
