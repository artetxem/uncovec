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

import argparse
import numpy as np
import sys


def read_embeddings(file, dtype='float32'):
    header = file.readline().split(' ')
    count = int(header[0])
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype)
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        words.append(word)
        matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
    return words, matrix


def write_embeddings(words, matrix, file):
    print('%d %d' % matrix.shape, file=file)
    for i in range(len(words)):
        print(words[i] + ' ' + ' '.join(['%.6g' % x for x in matrix[i]]), file=file)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Post-process word embeddings as described in our paper (https://arxiv.org/abs/1809.02094)')
    parser.add_argument('alpha', type=float, help='the alpha parameter (see paper)')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the input embeddings (defaults to stdin)')
    parser.add_argument('-o', '--output', default=sys.stdout.fileno(), help='the output embeddings (defaults to stdout)')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    args = parser.parse_args()

    # Read input embeddings
    with open(args.input, encoding=args.encoding, errors='surrogateescape') as f:
        words, x = read_embeddings(f)

    # Learn transformation
    l, q = np.linalg.eigh(x.T.dot(x))
    w = q*(l**args.alpha)

    # Write transformed embeddings
    with open(args.output, mode='w', encoding=args.encoding, errors='surrogateescape') as f:
        write_embeddings(words, x.dot(w), f)


if __name__ == '__main__':
    main()
