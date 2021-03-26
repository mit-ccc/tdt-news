# topic detection and tracking

## fine-tuning sentence BERT

Check `entity-bert` folder

## weight similarity model

### SVM-rank
1. generate data for weight similarity model with `python generate_svm_data.py`
2. train SVM-rank models with `sh train_svm_rank.sh`

### linear SVM with triplet loss

1. generate data with `python generate_triplet_data.py`
2. train SVM model with `python train_svm_triplet.py`

## cluster creation model

### SVM-liblinear
1. generate data for weight similarity model with `python generate_svm_data.py`
2. train SVM model with `python train_svm_merge.py`

### NN with L-BFGS

1. generate data for weight similarity model with `python generate_nn_data.py`
2. train LBFGS model with `python train_lbfgs.py`

## clustering

For each experiment, we create a new folder in `svm_en_data/output` to store the output results. 
We use the bash scripts for different experiments
1. `sh run_tfidf.sh` for experiments using TF-IDF features
2. `sh run_bert_tfidf.sh` for experiments combining TF-IDF and BERT features


## evaluation

run `python evaluate_model_output.py` after changing the output folder. 