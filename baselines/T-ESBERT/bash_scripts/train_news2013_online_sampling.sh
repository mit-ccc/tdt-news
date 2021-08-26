python run_retrospective_clustering.py --cluster_algorithm agg_average --features bert \
    --input_folder ../output/exp_sbert_news2013_ep10_mgn2.0_btch32_norm1.0_max_seq_230_sample_random

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python run_retrospective_clustering.py --cluster_algorithm agg_average --features bert \
        --input_folder ../output/exp_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
done 

python run_retrospective_clustering.py --cluster_algorithm agg_average --features bert \
    --input_folder ../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss



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

export CUDA_VISIBLE_DEVICES=1
# extract dense vectors from the test data
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    echo "epochnum", ${epochnum}
    python extract_features.py --dataset_name news2013 \
        --model_path ./output/exp_sbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/model_ep${epochnum}.pt
done

cd retrospective-tdt

# run retrospective TDT with the HDBSCAN algorithm
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    echo ${epochnum}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 --algorithm boruvka_kdtree \
    --input_folder ../output/exp_sbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
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

export CUDA_VISIBLE_DEVICES=2
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    echo "epochnum", ${epochnum}
    python extract_features.py --dataset_name news2013 \
        --model_path ./output/exp_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/model_ep${epochnum}.pt
done

cd retrospective-tdt

# run retrospective TDT with the HDBSCAN algorithm
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    echo ${epochnum}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 --algorithm boruvka_kdtree \
    --input_folder ../output/exp_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
done

#####################################################################
### Sin-PE-E-SBERT : News2013 -- online triplets
#####################################################################

export CUDA_VISIBLE_DEVICES=2
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} \
        --time_module sin_PE \
        --dataset_name news2013 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --time_encoding day
done

export CUDA_VISIBLE_DEVICES=3
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python extract_features.py --dataset_name news2013 \
    --model_path ./output/exp_pos2vec_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/model_ep${epochnum}.pt
done

cd retrospective-tdt

# run retrospective TDT with the HDBSCAN algorithm
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    echo ${epochnum}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 --algorithm boruvka_kdtree \
    --input_folder ../output/exp_pos2vec_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
done


# compare against other time-text fusion methods
export CUDA_VISIBLE_DEVICES=2
for fuse_method in additive additive_selfatt_pool additive_concat_selfatt_pool
do
    python train_pos2vec_esbert.py --num_epochs 2 \
        --dataset_name news2013 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method ${fuse_method} \
        --time_encoding day
done

for fuse_method in additive additive_selfatt_pool additive_concat_selfatt_pool
do
    python extract_features.py --dataset_name news2013 \
    --model_path ./output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_${fuse_method}_random_sample_BatchHardTripletLoss/model_ep2.pt
done


# set the file back to `test_data_reanchoring` and `test_sent_embeds_reanchoring`
for fuse_method in additive additive_selfatt_pool selfatt_pool additive_concat_selfatt_pool
do
    echo ${fuse_method}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 --algorithm boruvka_kdtree \
    --input_folder ../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_${fuse_method}_random_sample_BatchHardTripletLoss
done


# re-train for the second epoch
python extract_features.py --dataset_name news2013 \
        --model_path ./output/exp_sbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/model_ep2.pt

python extract_features.py --dataset_name news2013 \
        --model_path ./output/exp_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/model_ep2.pt

export CUDA_VISIBLE_DEVICES=1
python train_pos2vec_esbert.py --num_epochs 2 \
    --dataset_name news2013 \
    --max_seq_length 230 \
    --train_batch_size 32 \
    --fuse_method selfatt_pool \
    --time_encoding day


python extract_features.py --dataset_name news2013 \
    --model_path ./output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/model_ep2.pt

cd retrospective-tdt

python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 --algorithm boruvka_kdtree \
    --input_folder ../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss


#####################################################################
### Learned-PE-E-SBERT : News2013 -- online triplets
#####################################################################
export CUDA_VISIBLE_DEVICES=1
for fuse_method in additive additive_selfatt_pool additive_concat_selfatt_pool
do
    python train_pos2vec_esbert.py --num_epochs 2 \
        --time_module learned_PE \
        --dataset_name news2013 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method ${fuse_method} \
        --time_encoding day
done

for fuse_method in additive additive_selfatt_pool additive_concat_selfatt_pool
do
    python extract_features.py --dataset_name news2013 \
    --model_path ./output/exp_learned_pe_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_${fuse_method}_random_sample_BatchHardTripletLoss/model_ep2.pt
done

cd retrospective-tdt

for fuse_method in additive additive_selfatt_pool selfatt_pool additive_concat_selfatt_pool
do
    echo ${fuse_method}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 --algorithm boruvka_kdtree \
    --input_folder ../output/exp_learned_pe_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_${fuse_method}_random_sample_BatchHardTripletLoss
done

# epochs
export CUDA_VISIBLE_DEVICES=3
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} \
        --time_module learned_PE \
        --dataset_name news2013 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --time_encoding day
done

export CUDA_VISIBLE_DEVICES=4
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python extract_features.py --dataset_name news2013 \
    --model_path ./output/exp_learned_pe_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/model_ep${epochnum}.pt
done

cd retrospective-tdt

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    echo ${fuse_method}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 --algorithm boruvka_kdtree \
    --input_folder ../output/exp_learned_pe_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
done

#####################################################################
### Date2vec-PE-E-SBERT : News2013 -- online triplets
#####################################################################
export CUDA_VISIBLE_DEVICES=2
for freeze_time in 0 1 
do
    python train_date2vec_esbert.py --num_epochs 2 \
        --dataset_name news2013 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --freeze_time_module ${freeze_time}
done

python extract_features.py --dataset_name news2013 \
    --model_path ./output/exp_date2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/model_ep2.pt

python extract_features.py --dataset_name news2013 \
    --model_path ./output/exp_date2vec_fronzen_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/model_ep2.pt

cd ../retrospective-tdt

for exp_name in date2vec date2vec_fronzen
do
    echo ${exp_name}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 --algorithm boruvka_kdtree \
    --input_folder ../output/exp_${exp_name}_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_${fuse_method}_random_sample_BatchHardTripletLoss
done

# more epochs 
export CUDA_VISIBLE_DEVICES=2
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_date2vec_esbert.py --num_epochs ${epochnum} \
        --dataset_name news2013 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --freeze_time_module 0
done

cd ..

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
python extract_features.py --dataset_name news2013 \
    --model_path ./output/exp_date2vec_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/model_ep${epochnum}.pt
done

for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    echo ${epochnum}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 --algorithm boruvka_kdtree \
    --input_folder ../output/exp_date2vec_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
done


#####################################################################
### Sin-PE-E-SBERT : News2013 -- online triplets -- other online sampling strategies
#####################################################################

export CUDA_VISIBLE_DEVICES=2
for loss_func in BatchHardSoftMarginTripletLoss BatchSemiHardTripletLoss BatchAllTripletLoss
do
    python train_pos2vec_esbert.py --num_epochs 2 \
        --time_module sin_PE \
        --dataset_name news2013 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --time_encoding day \
        --loss_function ${loss_func}
done

for loss_func in BatchHardSoftMarginTripletLoss BatchSemiHardTripletLoss BatchAllTripletLoss
do
    python extract_features.py --dataset_name news2013 \
    --model_path ./output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_${loss_func}/model_ep2.pt
done

cd retrospective-tdt

# run retrospective TDT with the HDBSCAN algorithm
for loss_func in BatchHardTripletLoss BatchHardSoftMarginTripletLoss BatchSemiHardTripletLoss BatchAllTripletLoss
do
    echo ${loss_func}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 --algorithm boruvka_kdtree \
    --input_folder ../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_${loss_func}
done

#####################################################################
### Sin-PE-E-SBERT : News2013 -- offline triplets -- EPEN_triplets EPHN_triplets HPHN_triplets HPEN_triplets
#####################################################################

export CUDA_VISIBLE_DEVICES=1
for triple_mode in EPEN_triplets EPHN_triplets HPHN_triplets HPEN_triplets
do
    python train_pos2vec_esbert.py --num_epochs 2 \
        --time_module sin_PE \
        --dataset_name news2013 \
        --max_seq_length 128 \
        --train_batch_size 16 \
        --fuse_method selfatt_pool \
        --time_encoding day \
        --sample_method offline \
        --offline_triplet_mode ${triple_mode}
done

for triple_mode in EPEN_triplets EPHN_triplets HPHN_triplets HPEN_triplets
do
    python extract_features.py --dataset_name news2013 \
    --model_path ./output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch16_norm1.0_max_seq_128_fuse_selfatt_pool_offline_sample_offline_${triple_mode}/model_ep2.pt
done

cd retrospective-tdt

# run retrospective TDT with the HDBSCAN algorithm
for triple_mode in EPEN_triplets EPHN_triplets HPHN_triplets HPEN_triplets
do
    echo ${triple_mode}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 --algorithm boruvka_kdtree \
    --input_folder ../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch16_norm1.0_max_seq_128_fuse_selfatt_pool_offline_sample_offline_${triple_mode}
done

#####################################################################
### Sin-PE-E-SBERT : News2013 -- time granularity -- online
#####################################################################

export CUDA_VISIBLE_DEVICES=4
for time_code in hour 2day 3day 4day week month
do
    python train_pos2vec_esbert.py --num_epochs 2 \
        --time_module sin_PE \
        --dataset_name news2013 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --time_encoding ${time_code}
done

for time_code in hour 2day 3day 4day week month
do
    python extract_features.py --dataset_name news2013 \
    --model_path ./output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_${time_code}/model_ep2.pt
done

cd retrospective-tdt

# run retrospective TDT with the HDBSCAN algorithm
for time_code in hour 2day 3day 4day week month
do
    echo ${time_code}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 7 --min_samples 3 --algorithm boruvka_kdtree \
    --input_folder ../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_${time_code}
done
