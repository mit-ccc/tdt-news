# models with batch size 64 (working...)

# python train_time_esbert.py --num_epochs 10 --max_seq_length 128

# esbert
# python evaluate_entity_models.py \
# --model_path ./output/exp_esbert_ep2_mgn2.0_btch64_norm1.0_max_seq_128/esbert_model_ep2.pt

# python evaluate_entity_models.py \
# --model_path ./output/exp_esbert_ep5_mgn2.0_btch64_norm1.0_max_seq_128/esbert_model_ep5.pt

# python evaluate_entity_models.py \
# --model_path ./output/exp_esbert_ep10_mgn2.0_btch64_norm1.0_max_seq_128/esbert_model_ep10.pt


#time_esbert
# python evaluate_entity_models.py \
# --model_path ./output/exp_time_esbert_ep2_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool/time_esbert_model_ep2.pt

# python evaluate_entity_models.py \
# --model_path ./output/exp_time_esbert_ep5_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool/time_esbert_model_ep5.pt

# python evaluate_entity_models.py \
# --model_path ./output/exp_time_esbert_ep10_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool/time_esbert_model_ep10.pt

# python evaluate_entity_models.py \
# --model_path ./output/exp_time_esbert_ep2_mgn2.0_btch64_norm1.0_max_seq_128_fuse_concat_pool/time_esbert_model_ep2.pt

# python evaluate_entity_models.py \
# --model_path ./output/exp_time_esbert_ep2_mgn2.0_btch64_norm1.0_max_seq_128_fuse_concat_pool_linear/time_esbert_model_ep2.pt

#time_esbert for different loss and sampling techniques

python evaluate_entity_models.py \
--model_path ./output/exp_time_esbert_ep5_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardSoftMarginTripletLoss/time_esbert_model_ep5.pt

python evaluate_entity_models.py \
--model_path ./output/exp_time_esbert_ep5_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchSemiHardTripletLoss/time_esbert_model_ep5.pt

python evaluate_entity_models.py \
--model_path ./output/exp_time_esbert_ep5_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchAllTripletLoss/time_esbert_model_ep5.pt

python evaluate_entity_models.py \
--model_path ./output/exp_time_esbert_ep5_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_regular_sample/time_esbert_model_ep5.pt

python evaluate_entity_models.py \
--model_path ./output/exp_time_esbert_ep5_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_regular_sample_BatchHardSoftMarginTripletLoss/time_esbert_model_ep5.pt

python evaluate_entity_models.py \
--model_path ./output/exp_time_esbert_ep5_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_regular_sample_BatchSemiHardTripletLoss/time_esbert_model_ep5.pt

python evaluate_entity_models.py \
--model_path ./output/exp_time_esbert_ep5_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_regular_sample_BatchAllTripletLoss/time_esbert_model_ep5.pt


