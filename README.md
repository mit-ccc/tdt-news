# Topic Detection and Tracking (TDT) with Time-Aware Document Embeddings

[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc/4.0/)

## Getting started

### Installation

```The following versions are important
pip install -r requirements.txt
```

## Pre-trained Model & Data

### Pre-trained SBERT

```download pre-trained base-nli-stsb-mean-tokens BERT model
cd scripts
sh download_pretrained_bert.sh
```


### News2013 Data

- download raw data: `sh download_data.sh` (stored in `./dataset/`)
- download processed data in pickle format: `train_dev.pickle` and `test.pickle` [here](https://drive.google.com/drive/u/1/folders/1JCm2S9euC2AhyP9_IFcnMmUZN3tGG9nF) and put into `./dataset/`
    - NOTE: you can check 'preprocessing/extract-entities.ipynb' to see how entities are extracted with spacy

## Training

### Fine-tuning T-E-BERT

- check `bash_scripts/finetune_news2013_bert.sh`

### Retrospective TDT 

- read the `README.md` inside `retrospective-tdt`
- check `sh bash_scripts/train_news2013_offline.sh`

### Online TDT 

- read the `README.md` inside `retrospective-tdt`
- check `sh bash_scripts/train_news2013_online.sh`

## Contact
Hang Jiang (hjian42@mit.edu)

## Acknowledgement

This code is developed on the following two github projects: [sentence-transformers](https://github.com/UKPLab/sentence-transformers) and [news-clustering](https://github.com/Priberam/news-clustering). 
