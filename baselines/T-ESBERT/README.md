## dependencies

```
pip install transformers==3.5.0
pip install torch==1.6.0
pip install -U liblinear-official
pip install smote_variants
pip install imbalanced_databases
```

## Experiments

```set up environment
ssh -L 8000:localhost:8000 hjian42@matlaber7.media.mit.edu
docker exec -it hjian42-tdt-1 bash
sudo -iu hjian42
cd /mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/
export CUDA_VISIBLE_DEVICES=5
```

1. train SBERT models (saved in `output/exp_xxx/`):
    - SBERT: `train_sbert.py --num_epochs 2/5/10`
    - entity sBERT: `train_esbert.py --num_epochs 2/5/10`
    - time-entity sBERT: `train_time_esbert.py --num_epochs 2/5/10`
2. evaluate the models on the EventSim Task
    - pre-trained sBERT baseline: `python evaluate_sbert.py --model_path bert-base-nli-stsb-mean-tokens`
    - sBERT trained models: `python evaluate_sbert.py --model_path ./output/exp_sbert_ep2_mgn2.0_btch8_norm1.0_max_seq_512` (just the folder name)
    - entity sBERT: `python evaluate_entity_models.py --model_path ./output/exp_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_512/esbert_model_ep9.pt`
    - entity-time sBERT: `python evaluate_entity_models.py --model_path ./output/exp_time_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_512/time_esbert_model_ep9.pt`
3. tune the TDT pipeline:
    - `sh tune_pipeline.sh`