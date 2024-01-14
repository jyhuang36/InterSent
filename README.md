# InterSent
Code for our paper [Bridging Continuous and Discrete Spaces: Interpretable Sentence Representation Learning via Compositional Operations](https://arxiv.org/pdf/2305.14599.pdf) at EMNLP 2023

## Requirements
* transformers == 4.18.0
* pytorch-lightning == 1.6.1

## Data

* [ParaNMT](https://drive.google.com/file/d/19NQ87gEFYu3zOIp_VNYQZgmnwRuSIyJd/view)
* [Discofuse](https://github.com/google-research-datasets/discofuse) Wikipedia balanced portion
* [Wikisplit](https://github.com/google-research-datasets/wiki-split)
* [Google Sentence Compression](https://github.com/google-research-datasets/sentence-compression)
The data folder should have a similar structure as the following:
```
└── data 
    └── paranmt
        └── para-nmt-5m-processed.txt
    └── discofuse
        ├── discofuse-train-balanced.txt
        └── discofuse-valid-balanced.txt
        └── discofuse-test-balanced.txt
    └── wikisplit
        ├── wikisplit-train.txt
        └── wikisplit-valid.txt
        └── wikisplit-test.txt
    └── google
        ├── sent-comp-train.txt
        └── sent-comp-test.txt
```
## Training

To train InterSent from scratch, run the following:
```
bash train.sh
```

## Evaluation
To evaluate InterSent on interpretability, run the following with your checkpoint path:
```
bash test.sh
```
To evaluate InterSent on STS, run the following with your checkpoint path:
```
bash stseval.sh
```



