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


def save_sentences(sentences, filename, encoding):
    f = open(filename, mode='w', encoding=encoding, errors='surrogateescape')
    for tokens in sentences:
        print(' '.join(tokens), file=f)
    f.close()


def save_labels(labels, filename, encoding):
    f = open(filename, mode='w', encoding=encoding, errors='surrogateescape')
    for label in labels:
        print(label, file=f)
    f.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess a STS dataset')
    parser.add_argument('input_dir', help='input directory with the CSV files')
    parser.add_argument('output_dir', help='output directory in which to save the TXT files')
    parser.add_argument('embeddings', help='the word embeddings')
    parser.add_argument('--keep_stopwords', action='store_true', help='do not remove stopwords')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input (defaults to utf-8)')
    args = parser.parse_args()

    # Read embeddings
    f = open(args.embeddings, encoding=args.encoding, errors='surrogateescape')
    words, emb = utils.read_embeddings(f)

    # Build word to index map
    word2ind = {word: i for i, word in enumerate(words)}

    for dataset in ('train', 'dev', 'test'):
        # Read data
        src, trg, ref = utils.read_data(open('{0}/sts-{1}.csv'.format(args.input_dir, dataset), encoding=args.encoding, errors='surrogateescape'))

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

        # Save output files
        save_sentences(src, '{0}/src-{1}.txt'.format(args.output_dir, dataset), args.encoding)
        save_sentences(trg, '{0}/targ-{1}.txt'.format(args.output_dir, dataset), args.encoding)
        save_labels(ref, '{0}/label-{1}.txt'.format(args.output_dir, dataset), args.encoding)


if __name__ == '__main__':
    main()
