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

### Citing

If you use code or ideas from this project, please cite:

```
@misc{1704.07138,
Author = {Chris Hokamp and Qun Liu},
Title = {Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search},
Year = {2017},
Eprint = {arXiv:1704.07138},
}
```

### Project Structure


#### Core Abstractions


### Running Experiments

For PRIMT and Domain Adaptation experiments, the lexical constraints are stored in `*.json` files. 
The format is `[[[c1_t1, c1_t2], [c2_t1, c2_t2, c2_t3], ...], ...]`: 
Each constraint is a list of tokens, and each segment has a list of constraints. The length of the 
outer list in the `*.json` should be the same as number of segments in the source data. If there are no constraints for a
segment, there should be an empty list. 

#### Pick-Revise 

#### Domain-Adaptation 


### Performance

The current implementation is pretty slow, and it gets slower the more constraints you add :disappointed:. 
The GBS algorithm can be easily parallelized, because each cell in a column is independent of the others (see paper). 
However, implementing this requires us to make some assumptions about the underlying model, and would thus
limit the generality of the code base. If you have ideas about how to make things faster, please create an issue. 

### Features

Ensembling and weighted decoding for Nematus models


### Running Tests


