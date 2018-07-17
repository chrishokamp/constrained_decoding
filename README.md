## Lexically Constrained Decoding with Grid Beam Search

This project is a reference implementation of Grid Beam Search (GBS) as presented in [Lexically Constrained Decoding For Sequence Generation](https://arxiv.org/abs/1704.07138).

We provide two sample implementations of translation models -- one using our framework for
Neural Interactive Machine Translation, 
and another for models trained with [Nematus](https://github.com/rsennrich/nematus).

NMT models trained with Nematus model work out of the box. This project can also be used as a general-purpose 
ensembled decoder for Nematus models with or without constraints. 

### Quick Start

```
git clone https://github.com/chrishokamp/constrained_decoding.git
cd constrained_decoding
pip install -e .
```


#### Translating with a Nematus Model: A Full Example 

We assume you've already installed [Theano](http://deeplearning.net/software/theano/install_ubuntu.html) 

You need to install the **theano** branch of [Nematus](https://github.com/EdinburghNLP/nematus/tree/theano) 
```
git clone https://github.com/EdinburghNLP/nematus.git
cd nematus
git checkout theano
python setup.py install
``` 


Now download assets and run constrained translation
```
# CHANGE THIS TO A LOCAL PATH 
EXP_DIR=/data/mt_data/nematus_en-de_example

# Download one of the Edinburgh pre-trained Nematus models from WMT 2016
mkdir $EXP_DIR && cd $EXP_DIR
LANG_PAIR=en-de
wget -r --cut-dirs=2 -e robots=off -nH -np -R index.html* http://data.statmt.org/rsennrich/wmt16_systems/$LANG_PAIR/

cd $EXP_DIR
# get subword-nmt
git clone https://github.com/rsennrich/subword-nmt
SUBWORD=$EXP_DIR/subword-nmt

# get moses scripts for preprocessing
git clone https://github.com/marian-nmt/moses-scripts.git 
MOSES_SCRIPTS=$EXP_DIR/moses-scripts

SRC=en
TRG=de

# Download the WMT 16 En-De test data
TEST_DATA=$EXP_DIR/wmt_test
mkdir $TEST_DATA && cd $TEST_DATA
wget http://data.statmt.org/wmt16/translation-task/test.tgz
tar xvf test.tgz

# preprocess test data
# SRC
cat $TEST_DATA/test/newstest2016-ende-src.en.sgm | $MOSES_SCRIPTS/scripts/generic/input-from-sgm.perl | \
$MOSES_SCRIPTS/scripts/tokenizer/normalize-punctuation.perl -l $SRC | \
$MOSES_SCRIPTS/scripts/tokenizer/tokenizer.perl -l $SRC -penn | \
$MOSES_SCRIPTS/scripts/recaser/truecase.perl -model $EXP_DIR/$LANG_PAIR/truecase-model.$SRC | \
$SUBWORD/apply_bpe.py -c $EXP_DIR/$LANG_PAIR/$SRC$TRG.bpe > $TEST_DATA/newstest2016-$SRC.preprocessed

# we'll use this as the reference when computing BLEU scores
$MOSES_SCRIPTS/scripts/generic/input-from-sgm.perl < $TEST_DATA/test/newstest2016-ende-ref.de.sgm > $TEST_DATA/test/newstest2016-ende-ref.de

# extract a small sample from the test set
cat $TEST_DATA/newstest2016-$SRC.preprocessed | head -500 >  $TEST_DATA/newstest2016-$SRC.preprocessed.small


MODEL_DIR=$EXP_DIR/$LANG_PAIR
# Now open $MODEL_DIR/model.npz.json, editing the 'dictionaries' key to point to the full path of the dictionaries
# i.e. $EXP_DIR/$SRC_LANG-$TRG_LANG/vocab.en.json

# for example, mine looks like this:
#  "dictionaries": [
#    "/data/mt_data/nematus_en-de_example/en-de/vocab.en.json",
#    "/data/mt_data/nematus_en-de_example/en-de/vocab.de.json"
#  ],

# your path to `constrained_decoding`
GBS=~/projects/constrained_decoding

# run translation without constraints 
python $GBS/scripts/translate_nematus.py \
  -m $MODEL_DIR/model.npz \
  -c $MODEL_DIR/model.npz.json \
  -i $TEST_DATA/newstest2016-$SRC.preprocessed.small \
  --alignments_output $TEST_DATA/newstest2016-$SRC.preprocessed.small.alignments \
  > $TEST_DATA/newstest2016-$SRC.preprocessed.small.baseline_translated


# run translation with constraints 
CONSTRAINTS=$GBS/examples/nematus/wmt_16_en-de/sample.constraints.wmt-test.small.json

python $GBS/scripts/translate_nematus.py \
  -m $MODEL_DIR/model.npz \
  -c $MODEL_DIR/model.npz.json \
  -i $TEST_DATA/newstest2016-$SRC.preprocessed.small \
  --constraints $CONSTRAINTS \
  --alignments_output $TEST_DATA/newstest2016-$SRC.preprocessed.small.constrained.alignments \
  > $TEST_DATA/newstest2016-$SRC.preprocessed.small.constrained_translated

# Check BLEU Scores
# postprocess baseline
cat $TEST_DATA/newstest2016-$SRC.preprocessed.small.baseline_translated | sed 's/\@\@ //g' | \
$MOSES_SCRIPTS/scripts/recaser/detruecase.perl | \
$MOSES_SCRIPTS/scripts/tokenizer/detokenizer.perl -l $TRG \
> $TEST_DATA/newstest2016-$SRC.preprocessed.small.baseline_translated.postprocessed

# postprocess constrained
cat $TEST_DATA/newstest2016-$SRC.preprocessed.small.constrained_translated | sed 's/\@\@ //g' | \
$MOSES_SCRIPTS/scripts/recaser/detruecase.perl | \
$MOSES_SCRIPTS/scripts/tokenizer/detokenizer.perl -l $TRG \
> $TEST_DATA/newstest2016-$SRC.preprocessed.small.constrained_translated.postprocessed

# get BLEU scores
# we only used the first 500 lines 
cat $TEST_DATA/test/newstest2016-ende-ref.de | head -500 >  $TEST_DATA/test/newstest2016-ende-ref.de.small

# baseline
$MOSES_SCRIPTS/scripts/generic/multi-bleu.perl $TEST_DATA/test/newstest2016-ende-ref.de.small < $TEST_DATA/newstest2016-$SRC.preprocessed.small.baseline_translated.postprocessed

# with constraints
$MOSES_SCRIPTS/scripts/generic/multi-bleu.perl $TEST_DATA/test/newstest2016-ende-ref.de.small < $TEST_DATA/newstest2016-$SRC.preprocessed.small.constrained_translated.postprocessed

```


### Citing

If you use code or ideas from this project, please cite:

```
@InProceedings{hokamp-liu:2017:Long,
  author    = {Hokamp, Chris  and  Liu, Qun},
  title     = {Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {July},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {1535--1546},
  url       = {http://aclweb.org/anthology/P17-1141}
}
```


### Running Experiments

For PRIMT and Domain Adaptation experiments, the lexical constraints are stored in `*.json` files. 
The format is `[[[c1_t1, c1_t2], [c2_t1, c2_t2, c2_t3], ...], ...]`: 
Each constraint is a list of tokens, and each segment has a list of constraints. The length of the 
outer list in the `*.json` should be the same as number of segments in the source data. If there are no constraints for a
segment, there should be an empty list. 


### Performance

The current implementation is pretty slow, and it gets slower the more constraints you add :disappointed:. 
The GBS algorithm can be easily parallelized, because each cell in a column is independent of the others (see paper). 
However, implementing this requires us to make some assumptions about the underlying model, and would thus
limit the generality of the code base. If you have ideas about how to make things faster, please create an issue. 

### Features

Ensembling and weighted decoding for Nematus models


### Using the Prototype server

We provide a [very simple server](scripts/run_constrained_decoding_server.py) for convenience while prototyping. 





