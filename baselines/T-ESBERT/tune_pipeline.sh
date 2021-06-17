
### 1. extract features from the selected models

# python extract_features.py --model_path ./output/exp_sbert_ep1_mgn2.0_btch64_norm1.0_max_seq_128

# python extract_features.py --model_path ./output/exp_esbert_ep9_mgn2.0_btch64_norm1.0_max_seq_128/esbert_model_ep9.pt

# python extract_features.py --model_path ./output/exp_time_esbert_ep2_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/time_esbert_model_ep2.pt

# python extract_features.py --model_path ./output/exp_time_esbert_ep3_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_frozen/time_esbert_model_ep3.pt



### 2. generate SVM feature files

# python generate_svm_data.py --input_folder ./output/exp_sbert_ep1_mgn2.0_btch64_norm1.0_max_seq_128

# python generate_svm_data.py --input_folder ./output/exp_esbert_ep9_mgn2.0_btch64_norm1.0_max_seq_128

# python generate_svm_data.py --input_folder ./output/exp_time_esbert_ep2_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss

# python generate_svm_data.py --input_folder ./output/exp_time_esbert_ep3_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_frozen



### 3. train SVM-rank models

# input_folder=./output/exp_sbert_ep1_mgn2.0_btch64_norm1.0_max_seq_128
# for input_folder in ./output/exp_sbert_ep1_mgn2.0_btch64_norm1.0_max_seq_128 ./output/exp_esbert_ep9_mgn2.0_btch64_norm1.0_max_seq_128 ./output/exp_time_esbert_ep2_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss ./output/exp_time_esbert_ep3_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_frozen
# do
#     mkdir ${input_folder}/models
#     for c1 in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 10 100
#     do   
#         # tfidf
#         ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ${input_folder}/train_svm_rank_without_bert.dat ${input_folder}/models/weight_model_svmrank_tfidf_c${c1}.dat
#         # tfidf + BERT
#         ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ${input_folder}/train_svm_rank.dat ${input_folder}/models/weight_model_svmrank_tfidf_bert_c${c1}.dat
#     done
# done


### 4. train SVM-triplet weight models + svmlib merge models
# for input_folder in ./output/exp_sbert_ep1_mgn2.0_btch64_norm1.0_max_seq_128 ./output/exp_esbert_ep9_mgn2.0_btch64_norm1.0_max_seq_128 ./output/exp_time_esbert_ep2_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss ./output/exp_time_esbert_ep3_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_frozen
# do
#     python train_svm_triplet.py --input_folder ${input_folder}
#     python train_svm_merge.py --input_folder ${input_folder}
# done

# TO BE REMOVED
# for input_folder in ./output/exp_sbert_ep1_mgn2.0_btch64_norm1.0_max_seq_128 ./output/exp_esbert_ep9_mgn2.0_btch64_norm1.0_max_seq_128 
# for input_folder in ./output/exp_time_esbert_ep2_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss ./output/exp_time_esbert_ep3_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_frozen
# do
#     python train_svm_triplet.py --input_folder ${input_folder}
#     python train_svm_merge.py --input_folder ${input_folder}
# done


### 5. tune BERT-based TDT models with different weight + merge models

for input_folder in ./output/exp_sbert_ep1_mgn2.0_btch64_norm1.0_max_seq_128 ./output/exp_esbert_ep9_mgn2.0_btch64_norm1.0_max_seq_128 ./output/exp_time_esbert_ep2_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss ./output/exp_time_esbert_ep3_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_frozen
do
    mkdir ${input_folder}/predictions
    for c1 in 0.001 0.0001
    do
        for c2 in 10 0.1 1.0 
        do
            # SVM-rank-BERT + md_3
            python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svmrank_tfidf_bert_c${c1}.dat \
            --merge_model_dir dataset/md_3 \
            --output_filename ${input_folder}/predictions/svmrank_svmlib_tfidf_bert_smote_weightM_c${c1}_mergeM_md3

            # # SVM-rank-BERT + SVM-lib-SMOTE
            python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svmrank_tfidf_bert_c${c1}.dat \
            --merge_model_dir ${input_folder}/models/merge_model_tfidf_bert_smote_c${c2}.md \
            --output_filename ${input_folder}/predictions/svmrank_svmlib_tfidf_bert_smote_weightM_c${c1}_mergeM_c${c2}
            
            # SVM-triplet-BERT + md_3
            for lr in 0.1 0.001 0.001
	        do
                python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svm_triplet_tfidf_bert_c${c1}_lr${lr}_ep30.dat \
                --merge_model_dir dataset/md_3 \
                --output_filename ${input_folder}/predictions/svmtriplet_svmlib_tfidf_bert_smote_weightM_c${c1}_lr${lr}_ep30_mergeM_md3
            done

            # # SVM-triplet-BERT + SVM-lib-SMOTE
            for lr in 0.1 0.001 0.001
	        do
                python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svm_triplet_tfidf_bert_c${c1}_lr${lr}_ep30.dat \
                --merge_model_dir ${input_folder}/models/merge_model_tfidf_bert_smote_c${c2}.md \
                --output_filename ${input_folder}/predictions/svmtriplet_svmlib_tfidf_bert_smote_weightM_c${c1}_lr${lr}_ep30_mergeM_c${c2}
            done
        done
    done
done

