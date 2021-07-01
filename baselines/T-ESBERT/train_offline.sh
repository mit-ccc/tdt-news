# We train BERT in 4 settings (sBERT, E-sBERT, T-E-sBERT, T-E-sBERT with frozen Date2vec module)
# 1. train SBERT models (saved in `output/exp_{name}}/`)
# 2. evaluate the models on the EventSim Task
#  export CUDA_VISIBLE_DEVICES=1

#### sbert
# for epochnum in 1 2 3 4 5 6 7 8 9 10
# do
#     python train_sbert_offline.py --max_seq_length 128 --num_epochs ${epochnum} --train_batch_size 16 --sample_method offline --offline_triplet_data_path ./output/exp_sbert_ep1_mgn2.0_btch64_norm1.0_max_seq_128/train_dev_offline_triplets.pickle
# done

# for epochnum in 1 2 3 4 5 6 7 8 9 10
# do
#     python evaluate_sbert.py --use_saved_triplets --model_path ./output/exp_sbert_ep${epochnum}_mgn2.0_btch64_norm1.0_max_seq_128
# done


#### esbert
# export CUDA_VISIBLE_DEVICES=1
# for epochnum in 1 2 3 4 5 6 7 8 9 10
# do
#     python train_esbert.py --num_epochs ${epochnum} --max_seq_length 128 --train_batch_size 16 --sample_method offline --offline_triplet_data_path /mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/output/exp_esbert_ep7_mgn2.0_btch64_norm1.0_max_seq_128/train_dev_offline_triplets.pickle
# done

# for epochnum in 1 2 3 4 5 6 7 8 9 10
# do
#     python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_esbert_ep${epochnum}_mgn2.0_btch16_norm1.0_max_seq_128_sample_offline/esbert_model_ep${epochnum}.pt
# done


#### time esbert
# export CUDA_VISIBLE_DEVICES=2
# for epochnum in 1 2 3 4 5 6 7 8 9 10
# do
#     python train_time_esbert.py --num_epochs ${epochnum} --max_seq_length 128 --freeze_time_module 0 --train_batch_size 16 --sample_method offline --offline_triplet_data_path /mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/output/exp_time_esbert_ep3_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/train_dev_offline_triplets.pickle 
# done

# for epochnum in 1 2 3 4 5 6 7 8 9 10
# do
#     python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_time_esbert_ep${epochnum}_mgn2.0_btch16_norm1.0_max_seq_128_fuse_selfatt_pool_offline_sample_offline_sampling/time_esbert_model_ep${epochnum}.pt
# done


export CUDA_VISIBLE_DEVICES=5
# for sampling_name in EPEN_triplets EPHN_triplets HPHN_triplets HPEN_triplets
for sampling_name in EPEN_triplets EPHN_triplets HPHN_triplets HPEN_triplets
do
    # for epochnum in 1 2 3 4 5 6 7 8 9 10
    # do
    #     python train_time_esbert.py --num_epochs ${epochnum} --max_seq_length 128 --freeze_time_module 0 --train_batch_size 16 --sample_method ${sampling_name} --offline_triplet_data_path /mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/output/exp_time_esbert_ep3_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/train_dev_offline_triplets.pickle 
    # done

    for epochnum in 5 6 7 8 9
    do
        python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_time_esbert_ep${epochnum}_mgn2.0_btch16_norm1.0_max_seq_128_fuse_selfatt_pool_${sampling_name}_sample_offline/time_esbert_model_ep${epochnum}.pt
    done
done

### time esbert with frozen time module
# export CUDA_VISIBLE_DEVICES=1
# # for sampling_name in EPEN_triplets EPHN_triplets HPHN_triplets HPEN_triplets
# for sampling_name in EPEN_triplets
# do
#     for epochnum in 1 2 3 4 5 6 7 8 9 10
#     do
#         python train_time_esbert.py --num_epochs ${epochnum} --max_seq_length 128 --freeze_time_module 1 --train_batch_size 16 --sample_method ${sampling_name} --offline_triplet_data_path /mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/output/exp_time_esbert_ep2_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_frozen/train_dev_offline_triplets.pickle 
#     done

#     for epochnum in 1 2 3 4 5 6 7 8 9 10
#     do
#         python evaluate_entity_models.py --use_saved_triplets --model_path ./output/exp_time_esbert_ep${epochnum}_mgn2.0_btch16_norm1.0_max_seq_128_fuse_selfatt_pool_offline_sample_${sampling_name}_time_frozen/time_esbert_model_ep${epochnum}.pt
#     done
# done
