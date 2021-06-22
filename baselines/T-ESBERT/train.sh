# We train BERT in 4 settings (sBERT, E-sBERT, T-E-sBERT, T-E-sBERT with frozen Date2vec module)
# 1. train SBERT models (saved in `output/exp_{name}}/`)
# 2. evaluate the models on the EventSim Task


#### sbert
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_sbert.py --max_seq_length 128 --num_epochs ${epochnum}
done

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python evaluate_sbert.py --use_saved_triplets --model_path ./output/exp_sbert_ep${epochnum}_mgn2.0_btch64_norm1.0_max_seq_128
done


#### esbert
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_esbert.py --num_epochs ${epochnum} --max_seq_length 128 
done

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_esbert_ep${epochnum}_mgn2.0_btch64_norm1.0_max_seq_128/esbert_model_ep${epochnum}.pt
done


#### time esbert
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_time_esbert.py --num_epochs ${epochnum} --max_seq_length 128 --freeze_time_module 0
done

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_time_esbert_ep${epochnum}_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/time_esbert_model_ep${epochnum}.pt
done


### time esbert with frozen time module
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_time_esbert.py --num_epochs ${epochnum} --max_seq_length 128 --freeze_time_module 1
done

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_time_esbert_ep${epochnum}_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_frozen/time_esbert_model_ep${epochnum}.pt
done
