# TDT Pipeline with Time-Entity sentenceBERT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Getting started

### download data

- download raw data: `sh download_data.sh`
- download processed data in pickle format: `train_dev_data.pickle` and `test_data.pickle`

### pip install

- install packages `pip install -r requirements.txt`
    - The following versions are important

```
pip install transformers==3.5.0
pip install torch==1.6.0
pip install -U liblinear-official
pip install smote_variants
pip install imbalanced_databases
```

### docker environment (for Hang)

```enter my docker environment
ssh -L 8000:localhost:8000 hjian42@matlaber7.media.mit.edu
docker exec -it hjian42-tdt-1 bash
sudo -iu hjian42
cd /mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/
export CUDA_VISIBLE_DEVICES=5
```

## Finetune BERT embeddings

1. run `sh train.sh` to finetune BERT embeddings in 4 settings:
    - finetune BERT
    - evaluate BERT on the EventSim Task

## tune the TDT pipeline with the BERT embeddings:

1. `sh tune_pipeline.sh` to do the following:
    - extract BERT features from models
    - generate SVM data
    - train SVM models
    - tune the clustering algorithm