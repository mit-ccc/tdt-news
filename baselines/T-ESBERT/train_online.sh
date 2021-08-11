# We train BERT in 4 settings (sBERT, E-sBERT, T-E-sBERT, T-E-sBERT with frozen Date2vec module)
# 1. train SBERT models (saved in `output/exp_{name}}/`)
# 2. evaluate the models on the EventSim Task


#### sbert
export CUDA_VISIBLE_DEVICES=1
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_sbert.py --max_seq_length 256 --num_epochs ${epochnum} --train_batch_size 32
done

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python evaluate_sbert.py --use_saved_triplets --model_path ./output/exp_sbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_256
done


#### esbert
export CUDA_VISIBLE_DEVICES=2
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_esbert.py --num_epochs ${epochnum} --max_seq_length 256 --train_batch_size 32
done

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_256_sample_random/esbert_model_ep${epochnum}.pt
done

############################################################
#################### Date2Vec ##############################
############################################################

#### time esbert
export CUDA_VISIBLE_DEVICES=3
for epochnum in 1 2 3 4 5
do
    python train_time_esbert.py --num_epochs ${epochnum} --max_seq_length 230 --freeze_time_module 0 --train_batch_size 32
done

for epochnum in 1 2 3 4 5
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_time_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/time_esbert_model_ep${epochnum}.pt
done

### time esbert with frozen time module
export CUDA_VISIBLE_DEVICES=4
for epochnum in 1 2 3 4 5
do
    python train_time_esbert.py --num_epochs ${epochnum} --max_seq_length 230 --freeze_time_module 1 --train_batch_size 32
done

for epochnum in 1 2 3 4 5
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_time_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_frozen/time_esbert_model_ep${epochnum}.pt
done

############################################################
#################### Pos2Vec  ##############################
############################################################

### pos2vec esbert with concatenation+selfatt+pooling
export CUDA_VISIBLE_DEVICES=5
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} --max_seq_length 230 --train_batch_size 32 --fuse_method selfatt_pool --time_encoding day
done

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_no_shuffle/time_esbert_model_ep${epochnum}.pt
done

### pos2vec esbert with addition
export CUDA_VISIBLE_DEVICES=6
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} --max_seq_length 230 --train_batch_size 32 --fuse_method additive --time_encoding day
done

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss_no_shuffle/time_esbert_model_ep${epochnum}.pt
done


### pos2vec esbert with additive_selfatt_pool
export CUDA_VISIBLE_DEVICES=7
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} --max_seq_length 230 --train_batch_size 32 --fuse_method additive_selfatt_pool --time_encoding day
done

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_selfatt_pool_random_sample_BatchHardTripletLoss_no_shuffle/time_esbert_model_ep${epochnum}.pt
done


### pos2vec esbert with additive_concat_selfatt_pool
export CUDA_VISIBLE_DEVICES=4
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} --max_seq_length 230 --train_batch_size 32 --fuse_method additive_concat_selfatt_pool --time_encoding day
done

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_concat_selfatt_pool_random_sample_BatchHardTripletLoss_no_shuffle/time_esbert_model_ep${epochnum}.pt
done


###########################################################
#################### learnable PE #########################
###########################################################

### pos2vec esbert with concatenation+selfatt+pooling
export CUDA_VISIBLE_DEVICES=5
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_learned_pos2vec_esbert.py --num_epochs ${epochnum} --max_seq_length 230 --train_batch_size 32 --fuse_method selfatt_pool --time_encoding day
done

for epochnum in 1 2 3 4 5 6 7 8
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_learned_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/time_esbert_model_ep${epochnum}.pt
done
for epochnum in 1 2 3 4 5 6 7 8
do
    python extract_features.py --model_path ./output/exp_learned_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/time_esbert_model_ep${epochnum}.pt
done


export CUDA_VISIBLE_DEVICES=4
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_learned_pos2vec_esbert.py --num_epochs ${epochnum} --max_seq_length 230 --train_batch_size 32 --fuse_method additive --time_encoding day
done

for epochnum in 1 2 3 4 5 6 7 8
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_learned_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss/time_esbert_model_ep${epochnum}.pt
done
for epochnum in 1 2 3 4 5 6 7 8
do
    python extract_features.py --model_path ./output/exp_learned_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss/time_esbert_model_ep${epochnum}.pt
done


export CUDA_VISIBLE_DEVICES=3
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_learned_pos2vec_esbert.py --num_epochs ${epochnum} --max_seq_length 230 --train_batch_size 32 --fuse_method additive_selfatt_pool --time_encoding day
done

for epochnum in 1 2 3 4 5 6 7 8
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_learned_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_selfatt_pool_random_sample_BatchHardTripletLoss/time_esbert_model_ep${epochnum}.pt
done
for epochnum in 1 2 3 4 5 6 7 8
do
    python extract_features.py --model_path ./output/exp_learned_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_selfatt_pool_random_sample_BatchHardTripletLoss/time_esbert_model_ep${epochnum}.pt
done


export CUDA_VISIBLE_DEVICES=2
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_learned_pos2vec_esbert.py --num_epochs ${epochnum} --max_seq_length 230 --train_batch_size 32 --fuse_method additive_concat_selfatt_pool --time_encoding day
done

for epochnum in 1 2 3 4 5 6 7 8
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_learned_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_concat_selfatt_pool_random_sample_BatchHardTripletLoss/time_esbert_model_ep${epochnum}.pt
done
for epochnum in 1 2 3 4 5 6 7 8
do
    python extract_features.py --model_path ./output/exp_learned_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_concat_selfatt_pool_random_sample_BatchHardTripletLoss/time_esbert_model_ep${epochnum}.pt
done





export CUDA_VISIBLE_DEVICES=5
# 0.9722604309949564 for epoch 2
# for epochnum in 1 2 3
# do
#     python evaluate_sbert.py --use_saved_triplets --model_path ./output/exp_sbert_ep${epochnum}_mgn2.0_btch64_norm1.0_max_seq_128
# done

for epochnum in 1 2 3
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_esbert_ep${epochnum}_mgn2.0_btch64_norm1.0_max_seq_128/esbert_model_ep${epochnum}.pt
done

for epochnum in 1 2 3
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_time_esbert_ep${epochnum}_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/time_esbert_model_ep${epochnum}.pt
done

for epochnum in 1 2 3
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_time_esbert_ep${epochnum}_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_frozen/time_esbert_model_ep${epochnum}.pt
done


# whether the result is the same everytime
for epochnum in 1 2 3 4 5
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_256_sample_random/esbert_model_ep2.pt
done


export CUDA_VISIBLE_DEVICES=5
echo "concat-additive selfatt (DAY)"
for epochnum in 1 2 3 4
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_concat_selfatt_pool_random_sample_BatchHardTripletLoss/time_esbert_model_ep${epochnum}.pt
done

echo "concat selfatt"
for epochnum in 1 2 3 4
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_no_shuffle/time_esbert_model_ep${epochnum}.pt
done

echo "additive"
for epochnum in 1 2 3 4
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss_no_shuffle/time_esbert_model_ep${epochnum}.pt
done

echo "additive selfatt"
for epochnum in 1 2 3 4
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_selfatt_pool_random_sample_BatchHardTripletLoss_no_shuffle/time_esbert_model_ep${epochnum}.pt
done

echo "concat-additive selfatt"
for epochnum in 1 2 3 4
do
    python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_pos2vec_esbert_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_concat_selfatt_pool_random_sample_BatchHardTripletLoss_no_shuffle/time_esbert_model_ep${epochnum}.pt
done
