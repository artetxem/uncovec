#!/bin/bash
#
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

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA="$ROOT/data"

# Get the word analogy dataset from word2vec
mkdir -p "$DATA/analogy"
wget -q https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip -O "$DATA/word2vec.zip"
unzip -p "$DATA/word2vec.zip" word2vec/trunk/questions-words.txt > "$DATA/analogy/questions-words.txt"
rm -f "$DATA/word2vec.zip"

# Get the SimLex-999 and MEN word similarity/relatedness datasets
mkdir -p "$DATA/similarity"
wget -q https://fh295.github.io/SimLex-999.zip -O "$DATA/simlex.zip"
unzip -p "$DATA/simlex.zip" SimLex-999/SimLex-999.txt | cut -f1,2,4 | tail -n +2 > "$DATA/similarity/simlex.tsv"
rm -f "$DATA/simlex.zip"
wget -q https://staff.fnwi.uva.nl/e.bruni/resources/MEN.zip -O "$DATA/men.zip"
unzip -p "$DATA/men.zip" MEN/MEN_dataset_natural_form_full | tr ' ' '\t' > "$DATA/similarity/men.tsv"
rm -f "$DATA/men.zip"

# Get the STS-benchmark dataset
mkdir -p "$DATA/stsbenchmark"
wget -q http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz -O "$DATA/stsbenchmark.tar.gz"
tar -xzf "$DATA/stsbenchmark.tar.gz" -C "$DATA"
rm -f "$DATA/stsbenchmark.tar.gz"

# Get glove and fasttext pretrained embeddings
mkdir -p "$DATA/embeddings"
wget -q http://nlp.stanford.edu/data/glove.840B.300d.zip -O "$DATA/glove.zip"
wget -q https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip -O "$DATA/fasttext.zip"

(echo '2196017 300' && unzip -p "$DATA/glove.zip" glove.840B.300d.txt) > "$DATA/embeddings/glove.full.txt"
unzip -p "$DATA/fasttext.zip" crawl-300d-2M.vec > "$DATA/embeddings/fasttext.full.txt"
for emb in glove fasttext ; do
    (echo '200000 300' && tail -n +2 "$DATA/embeddings/$emb.full.txt" | head -200000) > "$DATA/embeddings/$emb.200k.txt"
    bzip2 "$DATA/embeddings/$emb.full.txt"
    rm -f "$DATA/$emb.zip"
done
