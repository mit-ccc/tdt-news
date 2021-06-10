## Experiments

```resources
docker exec -it hjian42-tdt-1 bash
sudo -iu hjian42
cd /mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/
export CUDA_VISIBLE_DEVICES=5

python train_sbert.py --num_epochs 2
python train_esbert.py --num_epochs 2
python train_time_esbert.py --num_epochs 2
```

1. train SBERT models (saved in `output/exp_xxx/`):
    - SBERT: `train_sbert.py --num_epochs 2/5/10`
    - entity sBERT: `train_esbert.py --num_epochs 2/5/10`
    - time-entity sBERT: `train_time_esbert.py --num_epochs 2/5/10`
2. evaluate the models on the EventSim Tasks
    - pre-trained sBERT baseline: `python evaluate_sbert.py --model_path bert-base-nli-stsb-mean-tokens`
    - sBERT trained models: `python evaluate_sbert.py --model_path ./output/exp_sbert_ep2_mgn2.0_btch8_norm1.0_max_seq_512`
    - entity sBERT: `python evaluate_entity_models.py --model_path ./output/exp_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_512/esbert_model_ep9.pt`
    - entity-time sBERT: `python evaluate_entity_models.py --model_path ./output/exp_time_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_512/time_esbert_model_ep9.pt`
2. extract features from the saved models (saved in `output/exp_xxx/`):
    - `extract_sbert_features.py` (TODO)
    - `extract_esbert_features.py` (TODO)
    - `extract_time_esbert_features.py`
3. generate feature files (.dat files) for SVMs:
    - `python generate_svm_data.py` (TODO: adjust input and output files)
4. training SVMs
    - `sh train_svm_rank.sh` and `sh train_svm_triplet.sh` for the weighting model
    - `python train_svm_merge.py` and `python train_lbfgs.py` for the merge model
5. run the clustering algorithm
    - `sh run_tfidf.sh` for experiments using TF-IDF features
    - `sh run_bert_tfidf.sh` for experiments combining TF-IDF and BERT features
