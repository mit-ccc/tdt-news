# Topic Detection and Tracking (TDT) with Time-Aware Document Embeddings

[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc/4.0/)

## Getting started

### install packages 

- install [svm_rank](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html) in this directory
- install the following packages
- for liblinear installation, you need to install gcc in Linux
    - `apt-get update`
    - `apt-get -y install gcc`
    - `sudo apt-get install g++`

```The following versions are important
pip install -U sentence-transformers
pip install transformers==3.5.0
pip install torch==1.6.0
pip install -U liblinear-official
pip install smote_variants
pip install imbalanced_databases
```


### download pre-trained BERT model

- download pre-trained sBERT model to `pretrained_bert` folder and convert its format to our model's format: 
    - `python download_pretrained_bert.py`
    - `mkdir ./pretrained_bert/exp_sbert_pretrained_max_seq_128`
    - `mv ./pretrained_bert/SBERT-base-nli-stsb-mean-tokens.pt ./pretrained_bert/exp_sbert_pretrained_max_seq_128/`
    - NOTE: we use "bert-base-nli-stsb-mean-tokens" BERT in our experiments

### download News2013 data

- download raw data: `sh download_data.sh` (stored in `./dataset/`)
- download processed data in pickle format: `train_dev.pickle` and `test.pickle` [here](https://drive.google.com/drive/u/1/folders/1JCm2S9euC2AhyP9_IFcnMmUZN3tGG9nF) and put into `./dataset/`
    - NOTE: you can check 'preprocessing/extract-entities.ipynb' to see how entities are extracted with spacy


## BERT finetuning + retrospective TDT 

- check `sh bash_scripts/train_news2013_offline.sh`


## Online TDT Task 

- after bert finetuning is done as shown above, run the following to get online results:
    - `sh bash_scripts/train_news2013_online.sh`
