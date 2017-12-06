# Bidirectional Attention Flow for Machine Comprehension

Bidirectional Attention Flow for Machine Comprehension, Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
https://arxiv.org/abs/1611.01603


This repo contains the implementation of Bi-Directional Attention Flow (BIDAF) network in PyTorch.

<img src="https://github.com/jojonki/BiDAF/blob/master/BiDAF.png?raw=true">

## Requirements
- PyTorch: 0.3
- tqdm (progress bar)

## Setup

```
$ ./download.sh
$ python -m squad.prepro    // build documents
```

## Training
```
$ python main.py --help
$ python main.py
```


## TODO
- [ ] confirm this model's performance, currently performance does not increasing. (see [#1](https://github.com/jojonki/BiDAF/issues/1) )
- [ ] clean main.py
