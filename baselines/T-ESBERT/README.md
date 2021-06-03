## Experiments

1. train SBERT models:
    - SBERT: TBD
    - entity-aware sBERT: `train_esbert.py`
    - time-entity-aware sBERT: `train_time_esbert.py`
2. extract features from the saved models:
    - `extract_esbert_features.py`
    - `extract_time_esbert_features.py`
3. switch to the other folder to generate features for SVMs