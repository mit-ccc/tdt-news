# TDT Pipeline with Time-Entity sentenceBERT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Getting started


### install packages 

- install [svm_rank](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html) in this directory
- install the following packages

```The following versions are important
pip install transformers==3.5.0
pip install torch==1.6.0
pip install -U liblinear-official
pip install smote_variants
pip install imbalanced_databases
```


### download News2013 data and pre-trained BERT model

- download raw data: `sh download_data.sh`
- download processed data in pickle format: `train_dev_data.pickle` and `test_data.pickle` [here](https://drive.google.com/drive/u/1/folders/1JCm2S9euC2AhyP9_IFcnMmUZN3tGG9nF) and put into `./dataset/`
    - NOTE: you can check 'preprocessing/extract-entities.ipynb' to see how entities are extracted with spacy
- download pre-trained sBERT model to `pretrained_bert` folder and convert its format to our model's format: `python download_pretrained_bert.py`
    - NOTE: we use "bert-base-nli-stsb-mean-tokens" BERT in our experiments


### docker environment (for Hang)

```enter my docker environment
ssh -L 8000:localhost:8000 hjian42@matlaber7.media.mit.edu
docker exec -it hjian42-tdt-1 bash
sudo -iu hjian42
cd /mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/
export CUDA_VISIBLE_DEVICES=5
```

## BERT finetuning + retrospective TDT 

- check `sh train_news2013.sh`




## Online TDT Task (TO BE UPDATED)
- coming soon

## Finetune BERT embeddings (online sampling) on the News dataset [IGNORE, TO BE UPDATED]

1. run `sh train.sh` to finetune BERT embeddings in 4 settings:
    - finetune BERT models (sBERT, E-sBERT, T-E-sBERT, T-E-sBERT with frozen Date2vec module)
    - evaluate BERT on the EventSim Task

2. run `sh train_offline.sh`
    - TODO: fix the bug for sBERT (training time too long)
    - TODO: make sBERT and T-E-sBERT share the same evaluation code


## Tune the TDT pipeline with the BERT embeddings [IGNORE, TO BE UPDATED]

1. `sh tune_pipeline.sh` to do the following:
    - extract BERT features from models
    - generate SVM data
    - train SVM models
    - tune the clustering algorithm
