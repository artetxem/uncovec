UncoVec
==============

This is an open source implementation of our word embedding post-processing and evaluation framework, described in the following paper:

Mikel Artetxe, Gorka Labaka, Iñigo Lopez-Gazpio, and Eneko Agirre. 2018. **[Uncovering divergent linguistic information in word embeddings with lessons for intrinsic and extrinsic evaluation](https://arxiv.org/pdf/1809.02094.pdf)**. In *Proceedings of the 22nd Conference on Computational Natural Language Learning (CoNLL 2018)*.

If you use this software for academic research, please cite the paper in question:
```
@inproceedings{artetxe2018conll,
  author    = {Artetxe, Mikel  and  Labaka, Gorka  and Lopez-Gazpio, Inigo  and  Agirre, Eneko},
  title     = {Uncovering divergent linguistic information in word embeddings with lessons for intrinsic and extrinsic evaluation},
  booktitle = {Proceedings of the 22nd Conference on Computational Natural Language Learning (CoNLL 2018)},
  month     = {October},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics}
}
```


Requirements
--------

If you just want to apply our proposed post-processing to your own embeddings, all you need is Python 3 with NumPy.

If you want to use the full evaluation framework, which tests embeddings in word analogy, word similarity/relatedness and semantic textual similarity for different values of the *alpha* parameter of our proposed post-processing, you will additionally need the following:
- Python 2 with the following dependencies:
  - NumPy
  - H5py
- Python 3 with the following dependencies:
  - Numpy
  - H5py
  - PyTorch (tested with v0.2)
  - NLTK
- A copy of [DAM-STS](https://github.com/lgazpio/DAM_STS) and [VecMap](https://github.com/artetxem/vecmap) at `third-party/`
- A copy of the evaluation datasets at `data/`

You will need to take care of the Python libraries yourself, but we provide the following scripts to automatically download the required datasets and the dependencies under `third-party/`:
```
./get-data.sh
./get-third-party.sh
```


Usage
--------

The following command applies the proposed post-processing to the given embeddings in word2vec text format:
```
python3 post-process.py ALPHA < INPUT.EMB.TXT > OUTPUT.EMB.TXT
```

Alternatively, you can run the full evaluation framework, which tests embeddings in word analogy, word similarity/relatedness and semantic textual similarity for different values of the *alpha* parameter (note that this requires an NVIDIA GPU with CUDA support):
```
./evaluate.sh EMBEDDINGS.TXT
```

Using the above script, you can reproduce the experiments reported in our paper as follows:
```
./evaluate.sh embeddings/glove.200k.txt
./evaluate.sh embeddings/fasttext.200k.txt
```


FAQ
-------

##### Why doesn't your script download and preprocess the word2vec embeddings used in your paper? How can I do that myself?

These embeddings are hosted in Google Drive and require a few clicks to download, which cannot be easily automatized using command line tools. In any case, you can do it manually as follows:
1. Download the embeddings from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).
2. Convert the embeddings from binary to text format. You can use [convertvec](https://github.com/marekrei/convertvec) for that.
3. Cut off the vocabulary as done in our experiments. You can use the following command for that:
```
(echo '200000 300' && tail -n +2 WORD2VEC.FULL.TXT | head -200000) > embeddings/word2vec.200k.txt
```

##### I am following the instructions in this README, but I am not getting the exact same numbers reported in your paper. Why is that?

You may get slightly different results for the following two reasons, but differences should be negligible and the general trends should be the same:
1. Different hardware and library versions might result in minor numerical variations in the underlying computations. These are generally imperceptible, but tend to be magnified by the stochastic nature of neural network training, which in this case affect the STS-DAM experiments. For that reason, we report the average across 10 runs for our STS-DAM experiments, which should minimize the effect of this type of variability.
2. In the case of MEN, the provided script downloads the official *natural form* version, whereas we used an in-house lemmatized version in our experiments. You should actually get slightly better results with the official version used in this public release. In any case, the general trends for different values of the *alpha* parameter are the same.


License
-------

Copyright (C) 2018, Mikel Artetxe

Licensed under the terms of the GNU General Public License, either version 3 or (at your option) any later version. A full copy of the license can be found in LICENSE.txt.