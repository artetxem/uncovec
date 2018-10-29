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

STS_RUNS=10

if [ "$#" -ne 1 ]; then
    echo "USAGE: evaluate.sh EMBEDDINGS.TXT"
    exit -1
fi


ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA="$ROOT/data"
DAM="$ROOT/third-party/dam-sts"
VECMAP="$ROOT/third-party/vecmap"
TMP=$(mktemp -d)


# Print header
echo -ne "embeddings\talpha\tanalogy-coverage\tanalogy-semantic\tanalogy-syntactic\tsimlex-coverage\tsimlex-spearman\tmen-coverage\tmen-spearman\tsts-centroid"
for ((i=1;i<=$STS_RUNS;i++)); do
    echo -ne "\tsts-dam-$i"
done
echo

for alpha in -1.00 -0.95 -0.90 -0.85 -0.80 -0.75 -0.70 -0.65 -0.60 -0.55 -0.50 -0.45 -0.40 -0.35 -0.30 -0.25 -0.20 -0.15 -0.10 -0.05 0.00 \
              0.05  0.10  0.15  0.20  0.25  0.30  0.35  0.40  0.45  0.50  0.55  0.60  0.65  0.70  0.75  0.80  0.85  0.90  0.95  1.00 ; do

    # Post-process embeddings
    python3 "$ROOT/post-process.py" $alpha -i "$1" -o "$TMP/embeddings.txt"
    echo -ne "$1\t$alpha\t"

    # Evaluate word analogy
    python3 "$VECMAP/eval_analogy.py" -i "$DATA/analogy/questions-words.txt" "$TMP/embeddings.txt" \
        | sed 's/[^0-9.]/ /g' | sed -E 's/\s+/\t/g' | cut -f2,4,5 | tr '\n' '\t'

    # Evaluate word similarity
    python3 "$VECMAP/eval_similarity.py" "$TMP/embeddings.txt" -i  "$DATA/similarity/simlex.tsv" "$DATA/similarity/men.tsv" \
        | head -2 | sed 's/[^0-9.]/ /g' | sed -E 's/\s+/\t/g' | cut -f2,4 | tr '\n' '\t'

    # Evaluate STS centroid
    python3 "$ROOT/sts/sts_centroid.py" "$DATA/stsbenchmark/sts-test.csv" "$TMP/embeddings.txt" | tr '\n' '\t'

    # Convert embeddings to glove format
    vocab_size=$(head -1 "$TMP/embeddings.txt" | cut -d' ' -f1)
    tail -n +2 "$TMP/embeddings.txt" > "$TMP/embeddings.glove.txt"

    # Convert STS dataset to DAM format
    python3 "$ROOT/sts/sts_preprocess.py" "$DATA/stsbenchmark" "$TMP" "$TMP/embeddings.txt" --keep_stopwords

    # Evaluate STS DAM
    for ((seed=0;seed<$STS_RUNS;seed++)); do
        mkdir "$TMP/dam"
        python2 "$DAM/preprocess_datasets/preprocess-STSBenchmark.py" \
            --srcfile "$TMP/src-train.txt" --targetfile "$TMP/targ-train.txt" --labelfile "$TMP/label-train.txt" \
            --srcvalfile "$TMP/src-dev.txt" --targetvalfile "$TMP/targ-dev.txt" --labelvalfile "$TMP/label-dev.txt" \
            --srctestfile "$TMP/src-test.txt" --targettestfile "$TMP/targ-test.txt" --labeltestfile "$TMP/label-test.txt" \
            --outputfile "$TMP/dam/data" --vocabsize $vocab_size --glove "$TMP/embeddings.glove.txt" --batchsize 8 --seed $seed >& /dev/null
        python2 "$DAM/preprocess_datasets/get_pretrain_vecs.py" \
            --glove "$TMP/embeddings.glove.txt" --outputfile "$TMP/embeddings.hdf5" \
            --dictionary "$TMP/dam/data.word.dict" --seed $seed >& /dev/null
        python3 "$DAM/DAM/DAM_STSBenchmark_TS.py" \
            --train_file "$TMP/dam/data-train.hdf5" \
            --dev_file "$TMP/dam/data-val.hdf5" \
            --test_file "$TMP/dam/data-test.hdf5" \
            --w2v_file "$TMP/embeddings.hdf5" \
            --log_dir "$TMP/dam/" \
            --log_fname test.log\
            --gpu_id 0 \
            --epoch 25 \
            --dev_interval 1 \
            --optimizer Adam \
            --lr 0.00005 \
            --hidden_size 2000 \
            --max_length -1 \
            --display_interval 500 \
            --weight_decay 0 \
            --dropout 0 \
            --bigrams \
            --model_path "$TMP/dam/" \
            --seed $seed \
            2>& 1 | tail -1 | sed 's/[^0-9.]/ /g' | sed -E 's/\s+/\t/g' | cut -f2 | tr '\n' '\t'
        rm -r "$TMP/dam"
    done
    echo

done
rm -r "$TMP"
