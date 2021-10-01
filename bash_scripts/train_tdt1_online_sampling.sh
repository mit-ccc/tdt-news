# TFIDF + Kmeans
for i in 1 2 3 4 5
do
    python run_retrospective_clustering.py --cluster_algorithm kmeans \
    --features tfidf \
    --random_seed ${i} \
    --gold_cluster_num 12 \
    --input_folder ../output/exp_sbert_tdt1_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
done

for min_cluster_size in 2 3 
do
    for min_samples in 2 3
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan --features tfidf \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_sbert_tdt1_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
    done
done



# TFIDF + GAC
python run_retrospective_clustering.py --cluster_algorithm agg_average \
    --features tfidf \
    --gold_cluster_num 12 \
    --input_folder ../output/exp_sbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random


# SBERT + GAC
for epochnum in 1 2 3 4 5
do
    python run_retrospective_clustering.py --cluster_algorithm agg_average \
    --features bert \
    --gold_cluster_num 12 \
    --input_folder ../output/exp_sbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
done


# E-SBERT + GAC
for epochnum in 1 2 3 4 5
do
    echo "epochnum", ${epochnum}
    python run_retrospective_clustering.py --cluster_algorithm agg_average \
    --features bert \
    --gold_cluster_num 12 \
    --input_folder ../output/exp_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
done

# Sin-PE-E-BERT + GAC
for epochnum in 1 2 3
do
    echo "epochnum", ${epochnum}
    python run_retrospective_clustering.py --cluster_algorithm agg_average \
    --features bert \
    --gold_cluster_num 12 \
    --input_folder ../output/exp_pos2vec_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_90day
done

for epochnum in 1 2 3
do
    echo "epochnum", ${epochnum}
    python run_retrospective_clustering.py --cluster_algorithm agg_average \
    --features bert \
    --gold_cluster_num 12 \
    --input_folder ../output/exp_pos2vec_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_month
done

for epochnum in 1 2 3
do
    echo "epochnum", ${epochnum}
    python run_retrospective_clustering.py --cluster_algorithm agg_average \
    --features bert \
    --gold_cluster_num 12 \
    --input_folder ../output/exp_pos2vec_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_180day
done


#####################################################################
### SBERT : TDT1 -- online training
#####################################################################
export CUDA_VISIBLE_DEVICES=1
# train 10 models from epoch 1 to 10
for epochnum in 1 2 3 4 5
do
    python train_entity_sbert_models.py --model_type sbert \
        --sample_method random \
        --dataset_name tdt1 \
        --num_epochs ${epochnum} \
        --max_seq_length 230 \
        --train_batch_size 32
done

# extract dense vectors from the test data
for epochnum in 1 2 3 4 5
do
    echo "epochnum", ${epochnum}
    python extract_features.py --dataset_name tdt1 \
        --model_path ./output/exp_sbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/model_ep${epochnum}.pt
done

cd retrospective-tdt

for min_cluster_size in 2 3 4 5 6 7 8 9 10
do
    for min_samples in 1 2 3 4 5 6 7 8 9 10
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_sbert_tdt1_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
    done
done

#####################################################################
### E-SBERT : TDT1 -- online training
#####################################################################
export CUDA_VISIBLE_DEVICES=2
# train 10 models from epoch 1 to 10
for epochnum in 1 2 3 4 5
do
    python train_entity_sbert_models.py --model_type esbert \
        --sample_method random \
        --dataset_name tdt1 \
        --num_epochs ${epochnum} \
        --max_seq_length 230 \
        --train_batch_size 32
done

# extract dense vectors from the test data
for epochnum in 1 2 3 4 5
do
    echo "epochnum", ${epochnum}
    python extract_features.py --dataset_name tdt1 \
        --model_path ./output/exp_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/model_ep${epochnum}.pt
done

cd retrospective-tdt

for min_cluster_size in 2 3 4 5 6 7 8 9 10
do
    for min_samples in 1 2 3 4 5 6 7 8 9 10
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_esbert_tdt1_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
    done
done

#####################################################################
### Sin-PE-E-SBERT : TDT1 -- online triplets (year)
#####################################################################

export CUDA_VISIBLE_DEVICES=1
for epochnum in 1 2 3
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} \
        --time_module sin_PE \
        --dataset_name tdt1 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --time_encoding year
done

for epochnum in 1 2 3
do
    python extract_features.py --dataset_name tdt1 \
    --model_path ./output/exp_pos2vec_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_year/model_ep${epochnum}.pt
done

cd retrospective-tdt

# run retrospective TDT with the HDBSCAN algorithm
for min_cluster_size in 2 3 4 5 6 7 8 9 10
do
    for min_samples in 1 2 3 4 5 6 7 8 9 10
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_pos2vec_esbert_tdt1_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_year
    done
done

#####################################################################
### Sin-PE-E-SBERT : TDT1 -- online triplets (180day)
#####################################################################

export CUDA_VISIBLE_DEVICES=2
for epochnum in 1 2 3
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} \
        --time_module sin_PE \
        --dataset_name tdt1 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --time_encoding 180day
done

for epochnum in 1 2 3
do
    python extract_features.py --dataset_name tdt1 \
    --model_path ./output/exp_pos2vec_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_180day/model_ep${epochnum}.pt
done

cd retrospective-tdt

# run retrospective TDT with the HDBSCAN algorithm
for min_cluster_size in 2 3 4 5 6 7 8 9 10
do
    for min_samples in 1 2 3 4 5 6 7 8 9 10
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_pos2vec_esbert_tdt1_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_180day
    done
done


#####################################################################
### Sin-PE-E-SBERT : TDT1 -- online triplets (90day)
#####################################################################

export CUDA_VISIBLE_DEVICES=3
for epochnum in 1 2 3
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} \
        --time_module sin_PE \
        --dataset_name tdt1 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --time_encoding 90day
done

for epochnum in 1 2 3
do
    python extract_features.py --dataset_name tdt1 \
    --model_path ./output/exp_pos2vec_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_90day/model_ep${epochnum}.pt
done

cd retrospective-tdt

# run retrospective TDT with the HDBSCAN algorithm
for min_cluster_size in 2 3 4 5 6 7 8 9 10
do
    for min_samples in 1 2 3 4 5 6 7 8 9 10
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_pos2vec_esbert_tdt1_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_90day
    done
done

#####################################################################
### Sin-PE-E-SBERT : TDT1 -- online triplets (2month)
#####################################################################

export CUDA_VISIBLE_DEVICES=1
for epochnum in 1 2 3
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} \
        --time_module sin_PE \
        --dataset_name tdt1 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --time_encoding 2month
done

for epochnum in 1 2 3
do
    python extract_features.py --dataset_name tdt1 \
    --model_path ./output/exp_pos2vec_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_2month/model_ep${epochnum}.pt
done

cd retrospective-tdt

# run retrospective TDT with the HDBSCAN algorithm
for min_cluster_size in 2 3 4 5 6 7 8 9 10
do
    for min_samples in 1 2 3 4 5 6 7 8 9 10
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_pos2vec_esbert_tdt1_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_2month
    done
done

#####################################################################
### Sin-PE-E-SBERT : TDT1 -- online triplets (month)
#####################################################################

export CUDA_VISIBLE_DEVICES=4
for epochnum in 1 2 3
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} \
        --time_module sin_PE \
        --dataset_name tdt1 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --time_encoding month
done

for epochnum in 1 2 3
do
    python extract_features.py --dataset_name tdt1 \
    --model_path ./output/exp_pos2vec_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_month/model_ep${epochnum}.pt
done

cd retrospective-tdt

# run retrospective TDT with the HDBSCAN algorithm
for min_cluster_size in 2 3 4 5 6 7 8 9 10
do
    for min_samples in 1 2 3 4 5 6 7 8 9 10
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_pos2vec_esbert_tdt1_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_month
    done
done

#####################################################################
### Sin-PE-E-SBERT : TDT1 -- online triplets (week)
#####################################################################

export CUDA_VISIBLE_DEVICES=5
for epochnum in 1 2 3
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} \
        --time_module sin_PE \
        --dataset_name tdt1 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --time_encoding week
done

for epochnum in 1 2 3
do
    python extract_features.py --dataset_name tdt1 \
    --model_path ./output/exp_pos2vec_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_week/model_ep${epochnum}.pt
done

cd retrospective-tdt

# run retrospective TDT with the HDBSCAN algorithm
for min_cluster_size in 2 3 4 5 6 7 8 9 10
do
    for min_samples in 1 2 3 4 5 6 7 8 9 10
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_pos2vec_esbert_tdt1_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_week
    done
done


#####################################################################
### Sin-PE-E-SBERT : TDT1 -- online triplets (day)
#####################################################################

export CUDA_VISIBLE_DEVICES=6
for epochnum in 1 2 3
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} \
        --time_module sin_PE \
        --dataset_name tdt1 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --time_encoding day
done

for epochnum in 1 2 3
do
    python extract_features.py --dataset_name tdt1 \
    --model_path ./output/exp_pos2vec_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_day/model_ep${epochnum}.pt
done

cd retrospective-tdt

# run retrospective TDT with the HDBSCAN algorithm
for min_cluster_size in 2 3 4 5 6 7 8 9 10
do
    for min_samples in 1 2 3 4 5 6 7 8 9 10
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_pos2vec_esbert_tdt1_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_day
    done
done


#####################################################################
### Date2vec-PE-E-SBERT : TDT1 -- online triplets
#####################################################################
export CUDA_VISIBLE_DEVICES=2
for epochnum in 1 2 3 4 5
do
    python train_date2vec_esbert.py --num_epochs ${epochnum} \
        --dataset_name tdt1 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --freeze_time_module 0
done

cd ..

for epochnum in 1 2 3 4 5
do
python extract_features.py --dataset_name tdt1 \
    --model_path ./output/exp_date2vec_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/model_ep${epochnum}.pt
done

cd retrospective-tdt

for epochnum in 1 2 3 4 5
do
    echo ${epochnum}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 3 --min_samples 3 --algorithm boruvka_kdtree \
    --input_folder ../output/exp_date2vec_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
done

for min_cluster_size in 2 3 4 5 6 7 8 9 10
do
    for min_samples in 1 2 3 4 5 6 7 8 9 10
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_date2vec_esbert_tdt1_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
    done
done




#####################################################################
### Learned-PE-E-SBERT : tdt1 -- online triplets
#####################################################################
export CUDA_VISIBLE_DEVICES=1
for epochnum in 1 2 3 4 5
do
    python train_pos2vec_esbert.py --num_epochs ${epochnum} \
        --time_module learned_PE \
        --dataset_name tdt1 \
        --max_seq_length 230 \
        --train_batch_size 32 \
        --fuse_method selfatt_pool \
        --time_encoding 90day
done

for epochnum in 1 2 3 4 5
do
    python extract_features.py --dataset_name tdt1 \
    --model_path ./output/exp_learned_pe_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_90day/model_ep${epochnum}.pt
done

cd retrospective-tdt

for epochnum in 1 2 3 4 5
do
    echo ${fuse_method}
    python run_retrospective_clustering.py --cluster_algorithm hdbscan \
    --min_cluster_size 3 --min_samples 3 --algorithm boruvka_kdtree \
    --input_folder ../output/exp_learned_pe_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_90day
done


for min_cluster_size in 2 3 4 5 6 7 8 9 10
do
    for min_samples in 1 2 3 4 5 6 7 8 9 10
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_learned_pe_esbert_tdt1_ep4_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_90day
    done
done


#####################################################################
### Sin-PE-E-SBERT : TDT1 -- online triplets (90day) -- other attention fusion 
#####################################################################

export CUDA_VISIBLE_DEVICES=4
for fuse_method in additive additive_selfatt_pool additive_concat_selfatt_pool
do
    for epochnum in 1 2 3
    do
        python train_pos2vec_esbert.py --num_epochs ${epochnum} \
            --time_module sin_PE \
            --dataset_name tdt1 \
            --max_seq_length 230 \
            --train_batch_size 32 \
            --fuse_method ${fuse_method} \
            --time_encoding 90day
    done
done

for fuse_method in additive additive_selfatt_pool additive_concat_selfatt_pool
do
    for epochnum in 1 2 3
    do
        python extract_features.py --dataset_name tdt1 \
        --model_path ./output/exp_pos2vec_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_${fuse_method}_random_sample_BatchHardTripletLoss_time_90day/model_ep${epochnum}.pt
    done
done

cd retrospective-tdt

# run retrospective TDT with the HDBSCAN algorithm
for fuse_method in additive additive_selfatt_pool additive_concat_selfatt_pool
do
    for epochnum in 1 2 3 4 5
    do
        echo ${fuse_method}, ${epochnum}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size 2 --min_samples 3 --algorithm boruvka_kdtree \
        --input_folder ../output/exp_pos2vec_esbert_tdt1_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_${fuse_method}_random_sample_BatchHardTripletLoss_time_90day
    done
done

# additive
for min_cluster_size in 2 3 4 5 6 7 8 9 10
do
    for min_samples in 1 2 3 4 5 6 7 8 9 10
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_pos2vec_esbert_tdt1_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss_time_90day
    done
done

# additive_selfatt_pool
for min_cluster_size in 2 3 4 5 6 7 8 9 10
do
    for min_samples in 1 2 3 4 5 6 7 8 9 10
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_pos2vec_esbert_tdt1_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_selfatt_pool_random_sample_BatchHardTripletLoss_time_90day
    done
done

# additive_concat_selfatt_pool
for min_cluster_size in 2 3 4 5 6 7 8 9 10
do
    for min_samples in 1 2 3 4 5 6 7 8 9 10
    do
        echo "min_cluster_size", ${min_cluster_size}, "min_samples", ${min_samples}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size ${min_cluster_size} --min_samples ${min_samples} --algorithm boruvka_kdtree \
        --input_folder ../output/exp_pos2vec_esbert_tdt1_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_concat_selfatt_pool_random_sample_BatchHardTripletLoss_time_90day
    done
done
