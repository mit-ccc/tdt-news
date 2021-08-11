#####################################################################
### SBERT : News2013 -- online training
#####################################################################
export CUDA_VISIBLE_DEVICES=0
# train 10 models from epoch 1 to 10
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_entity_sbert_models.py --model_type sbert \
        --sample_method random \
        --dataset_name news2013 \
        --num_epochs ${epochnum} \
        --max_seq_length 230 \
        --train_batch_size 32
done

# extract dense vectors from the test data
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    echo "epochnum", ${epochnum}
    python extract_features.py --dataset_name news2013 \
        --model_path ./output/exp_sbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/model_ep${epochnum}.pt
done

# run retrospective TDT with the HDBSCAN algorithm
for num_epoch in 1 2 3 4 5 6 7 8 9 10
do
    echo ${iter}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 \
    --input_folder ./output/exp_sbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
done

##############################################
### E-SBERT : News2013 -- online training
##############################################
export CUDA_VISIBLE_DEVICES=1
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_entity_sbert_models.py --model_type esbert \
        --sample_method random \
        --dataset_name news2013 \
        --num_epochs ${epochnum} \
        --max_seq_length 230 \
        --train_batch_size 32
done

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    echo "epochnum", ${epochnum}
    python extract_features.py --dataset_name news2013 \
        --model_path ./output/exp_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/model_ep${epochnum}.pt
done

# run retrospective TDT with the HDBSCAN algorithm
for num_epoch in 1 2 3 4 5 6 7 8 9 10
do
    echo ${iter}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 \
    --input_folder ./output/exp_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
done

#####################################################################
### T-E-SBERT : News2013 -- online triplets
#####################################################################

export CUDA_VISIBLE_DEVICES=2
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} \
        --dataset_name news2013 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --time_encoding day
done

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python extract_features.py --dataset_name news2013 \
    --model_path ./output/exp_pos2vec_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/model_ep${epochnum}.pt
done

# run retrospective TDT with the HDBSCAN algorithm
for num_epoch in 1 2 3 4 5 6 7 8 9 10
do
    echo ${iter}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 \
    --input_folder ./output/exp_pos2vec_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
done

