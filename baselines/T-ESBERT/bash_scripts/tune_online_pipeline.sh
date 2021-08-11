
####
# Tune TDT pipeline + BERT finetuned with online sampling settings
####

## 1. extract features from the selected models

# python extract_features.py --model_path ./output/exp_sbert_ep3_mgn2.0_btch32_norm1.0_max_seq_256

# python extract_features.py --model_path ./output/exp_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_256_sample_random/esbert_model_ep2.pt

# python extract_features.py --model_path ./output/exp_time_esbert_ep3_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/time_esbert_model_ep3.pt

# python extract_features.py --model_path ./output/exp_time_esbert_ep2_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_frozen/time_esbert_model_ep2.pt

# NEW ONES
# python extract_features.py --model_path ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss/time_esbert_model_ep2.pt

# python extract_features.py --model_path ./output/exp_pos2vec_esbert_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_selfatt_pool_random_sample_BatchHardTripletLoss/time_esbert_model_ep1.pt

# python extract_features.py --model_path ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/time_esbert_model_ep2.pt



for input_folder in ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
do
    mkdir ${input_folder}/models
    # ## generate SVM data
    # python generate_svm_data.py --input_folder ${input_folder}

    # for c1 in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 10 100
    # do
    #     ## training SVM-rank Models
    #     # tfidf
    #     ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ${input_folder}/train_svm_rank_without_bert.dat ${input_folder}/models/weight_model_svmrank_tfidf_c${c1}.dat
    #     # tfidf + BERT
    #     ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ${input_folder}/train_svm_rank.dat ${input_folder}/models/weight_model_svmrank_tfidf_bert_c${c1}.dat
    # done

    # ## trainingnSVM-triplet
    # python train_svm_triplet.py --input_folder ${input_folder}

    # ## training SVM-merge
    # python train_svm_merge.py --input_folder ${input_folder}

    ## tuning the TDT clustering algorithm
    # rm -rf ${input_folder}/predictions
    mkdir ${input_folder}/predictions
    for c1 in 0.1
    do
        # for c2 in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9
        for c2 in 0.1 100 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 10 100
        do
            # SVM-rank-BERT + md_3
            # python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svmrank_tfidf_bert_c${c1}.dat \
            # --merge_model_dir dataset/md_3 \
            # --output_filename ${input_folder}/predictions/svmrank_svmlib_tfidf_bert_smote_weightM_c${c1}_mergeM_md3

            # # SVM-rank-BERT + SVM-lib-SMOTE
            # python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svmrank_tfidf_bert_c${c1}.dat \
            # --merge_model_dir ${input_folder}/models/merge_model_tfidf_bert_smote_c${c2}.md \
            # --output_filename ${input_folder}/predictions/svmrank_svmlib_tfidf_bert_smote_weightM_c${c1}_mergeM_c${c2}

            # python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svmrank_tfidf_bert_c${c1}.dat \
            # --merge_model_dir ${input_folder}/models/merge_model_tfidf_bert_smote_c${c2}_b1.md \
            # --output_filename ${input_folder}/predictions/svmrank_svmlib_tfidf_bert_smote_weightM_c${c1}_mergeM_c${c2}
            
            # SVM-triplet-BERT + md_3
            # for lr in 0.1
            # do
            #     python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svm_triplet_tfidf_bert_c${c1}_lr${lr}_ep30.dat \
            #     --merge_model_dir dataset/md_3 \
            #     --output_filename ${input_folder}/predictions/svmtriplet_svmlib_tfidf_bert_smote_weightM_c${c1}_lr${lr}_ep30_mergeM_md3
            # done

            # # SVM-triplet-BERT + SVM-lib-SMOTE
            for lr in 0.1
            do
                # NO BERT
                # python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svm_triplet_tfidf_bert_c${c1}_lr${lr}_ep30.dat \
                # --merge_model_dir ${input_folder}/models/merge_model_tfidf_smote_c${c2}_b0.md \
                # --output_filename ${input_folder}/predictions/svmtriplet_svmlib_tfidf_smote_b0_weightM_c${c1}_lr${lr}_ep30_mergeM_c${c2}

                # python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svm_triplet_tfidf_bert_c${c1}_lr${lr}_ep30.dat \
                # --merge_model_dir ${input_folder}/models/merge_model_tfidf_smote_c${c2}_b1.md \
                # --output_filename ${input_folder}/predictions/svmtriplet_svmlib_tfidf_smote_b1_weightM_c${c1}_lr${lr}_ep30_mergeM_c${c2}

                ########## BERT + TFIDF + TIME
                # python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svm_triplet_tfidf_bert_c${c1}_lr${lr}_ep30_no_bias.dat \
                # --merge_model_dir ${input_folder}/models/merge_model_tfidf_bert_smote_c${c2}_b0.md \
                # --output_filename ${input_folder}/predictions/svmtriplet_svmlib_tfidf_bert_smote_b0_weightM_c${c1}_lr${lr}_ep30_non_bias_mergeM_c${c2} \
                # --data_path ${input_folder}/test_data.pickle

                # python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svm_triplet_tfidf_bert_c${c1}_lr${lr}_ep30.dat \
                # --merge_model_dir dataset/md_3 \
                # --output_filename ${input_folder}/predictions/svmtriplet_svmlib_tfidf_bert_smote_b0_weightM_c${c1}_lr${lr}_ep30_mergeM_md3 \
                # --data_path ${input_folder}/test_data.pickle

                ########## BERT + TFIDF - TIME
                python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svm_triplet_bert_ablate_time_c${c1}_lr${lr}_ep30_no_bias.dat \
                --merge_model_dir ${input_folder}/models/merge_model_bert_ablate_time_smote_c${c2}_b0.md \
                --output_filename ${input_folder}/predictions/svmtriplet_svmlib_bert_ablate_time_smote_b0_weightM_c${c1}_lr${lr}_ep30_mergeM_c${c2} \
                --data_path ${input_folder}/test_data.pickle \
                --weight_model_ii_file ./dataset/bert_ablate_time.ii

                ########## BERT - TFIDF + TIME
                # python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svm_triplet_bert_ablate_tfidf_c${c1}_lr${lr}_ep30_no_bias.dat \
                # --merge_model_dir ${input_folder}/models/merge_model_bert_ablate_tfidf_smote_c${c2}_b0.md \
                # --output_filename ${input_folder}/predictions/svmtriplet_svmlib_bert_ablate_tfidf_smote_b0_weightM_c${c1}_lr${lr}_ep30_mergeM_c${c2} \
                # --data_path ${input_folder}/test_data.pickle \
                # --weight_model_ii_file ./dataset/bert_ablate_tfidf.ii

                # (NOT WORKING WITH a BIAS)
                # python testbench.py --weight_model_dir ${input_folder}/models/weight_model_svm_triplet_tfidf_bert_c${c1}_lr${lr}_ep30.dat \
                # --merge_model_dir ${input_folder}/models/merge_model_tfidf_bert_smote_c${c2}_b1.md \
                # --output_filename ${input_folder}/predictions/svmtriplet_svmlib_tfidf_bert_smote_b1_weightM_c${c1}_lr${lr}_ep30_mergeM_c${c2} \
                # --data_path ${input_folder}/test_data.pickle

                ########### ONLY BERT
                # python testbench_bert.py --weight_model_dir ${input_folder}/models/weight_model_svm_triplet_pure_bert_c${c1}_lr${lr}_ep30.dat \
                # --merge_model_dir ${input_folder}/models/merge_model_pure_bert_smote_c${c2}_b1.md \
                # --output_filename ${input_folder}/predictions/svmtriplet_svmlib_pure_bert_smote_b1_weightM_c${c1}_lr${lr}_ep30_mergeM_c${c2} \
                # --data_path ${input_folder}/test_data.pickle

            done
        done
    done
done


# collect the performance from output files

# python evaluate_model_outputs.py --output_folder ./output/exp_sbert_ep3_mgn2.0_btch32_norm1.0_max_seq_256

# python evaluate_model_outputs.py --output_folder ./output/exp_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_256_sample_random

# python evaluate_model_outputs.py --output_folder ./output/exp_time_esbert_ep3_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss

# python evaluate_model_outputs.py --output_folder ./output/exp_time_esbert_ep2_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_frozen

# python evaluate_model_outputs.py --output_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss

# python evaluate_model_outputs.py --output_folder ./output/exp_pos2vec_esbert_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_selfatt_pool_random_sample_BatchHardTripletLoss

# python evaluate_model_outputs.py --output_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss

python evaluate_model_outputs.py --output_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss