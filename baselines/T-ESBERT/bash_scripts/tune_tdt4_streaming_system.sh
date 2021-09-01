export CUDA_VISIBLE_DEVICES=1
##########################################################################################################################################
### TFIDF + TIME : TDT4 -- online training - Miranda et al 2018
##########################################################################################################################################
feature_option=tfidf_time
# use ep4 because this is rarely the best fine-tuned model
input_folder=../output/exp_sbert_tdt4_ep4_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

# train svm_rank models
for c1 in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 10 100 1000 10000
do
    ## training SVM-rank Models
    mkdir ${input_folder}/${feature_option}/svm_rank_models
    ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ${input_folder}/${feature_option}/train_svm_rank.dat ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat
done

# train libSVM merge models
python train_svm_merge.py --input_folder ${input_folder}/${feature_option}/ --use_smote 0 --features ${feature_option}
python train_svm_merge.py --input_folder ${input_folder}/${feature_option}/ --use_smote 1 --features ${feature_option}

mkdir ${input_folder}/${feature_option}/predictions
mkdir ${input_folder}/${feature_option}/predictions/cross_validations
# 0.1 0.01 0.001 0.0001 0.00001 0.5 1.0 10
for c1 in 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10 100
do
    # for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
    for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
    do
            # cross validation
        echo "running corss validation"
        python testbench.py --use_cross_validation 1 \
        --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat \
        --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models/libSVM_c${c2}_b0.md \
        --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_b0_weightM_c${c1}_mergeM_c${c2} \
        --data_path ${input_folder}/test_bert.pickle \
        --weight_model_ii_file ./meta_features/tdt4/tfidf_time.ii
    done 
done

feature_option=tfidf_time
input_folder=../output/exp_sbert_tdt4_ep4_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c10.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models/libSVM_c0.01_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c10_mergeM_c0.01 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/tdt4/tfidf_time.ii

python evaluate_model_outputs.py --dataset_name tdt4 --prediction_path ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c10_mergeM_c0.01eng.out

##########################################################################################################################################
### TFIDF + TIME : TDT4 -- online training (SMOTE + SVM-Triplet + NN-LBFGS) - EACL paper 2021
##########################################################################################################################################
feature_option=tfidf_time_esbert
# use ep5 because this is rarely the best fine-tuned model
input_folder=../output/exp_sbert_tdt4_ep5_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

# train svm_rank models
for c1 in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 10 100 1000 10000
do
    ## training SVM-rank Models
    mkdir ${input_folder}/${feature_option}/svm_rank_models
    ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ${input_folder}/${feature_option}/train_svm_rank.dat ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat
done

python train_weight_model_svm_triplet.py --input_folder ${input_folder}/${feature_option}
# python train_weight_model_margin_ranking.py --input_folder ${input_folder}/${feature_option}

# generate additional paper for training merge model
for c1 in 0.0001 0.001 0.01 0.1 0.5 1.0 10 100 1000
do
    python generate_lbfgs_data.py --input_folder ${input_folder} --features ${feature_option} --weight_model ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat --weight_model_ii_file ./meta_features/tdt4/${feature_option}.ii
    # python generate_lbfgs_data.py --input_folder ${input_folder} --features ${feature_option} --weight_model ${input_folder}/${feature_option}/svm_triplet_weight_models/SVM_triplet_c10.dat --weight_model_ii_file ./meta_features/tdt4/tfidf_time.ii
    # python generate_lbfgs_data.py --input_folder ${input_folder} --features ${feature_option} --weight_model ${input_folder}/${feature_option}/svm_triplet_weight_models/SVM_triplet_c0.1.dat --weight_model_ii_file ./meta_features/tdt4/tfidf_time.ii
    # python generate_lbfgs_data.py --input_folder ${input_folder} --features ${feature_option} --weight_model ${input_folder}/${feature_option}/svm_triplet_weight_models/SVM_triplet_c0.5.dat --weight_model_ii_file ./meta_features/tdt4/tfidf_time.ii
    # python generate_lbfgs_data.py --input_folder ${input_folder} --features ${feature_option} --weight_model ${input_folder}/${feature_option}/svm_triplet_weight_models/SVM_triplet_c1.0.dat --weight_model_ii_file ./meta_features/tdt4/tfidf_time.ii
done 

# python train_merge_model_nn_lbfgs.py --data_path ${input_folder}/${feature_option}/train_lbfgs_raw_c0.1.dat
# python train_merge_model_nn_lbfgs.py --data_path ${input_folder}/${feature_option}/train_lbfgs_raw_c0.5.dat
# python train_merge_model_nn_lbfgs.py --data_path ${input_folder}/${feature_option}/train_lbfgs_raw_c1.0.dat



# # train libSVM merge models
python train_svm_merge.py --data_path ${input_folder}/${feature_option}/train_lbfgs_raw_c1000.dat --use_smote 0 --features ${feature_option}
python train_svm_merge.py --data_path ${input_folder}/${feature_option}/train_lbfgs_raw_c1000.dat --use_smote 1 --features ${feature_option}

mkdir ${input_folder}/${feature_option}/predictions
mkdir ${input_folder}/${feature_option}/predictions/cross_validations
#  0.00005 0.0001 0.0005 0.001 0.1 0.01 0.001
# for c1 in 0.1 0.5 1.0 10
# do
#     # for c2 in 0.6 0.7 0.8 0.9 1.0
#     for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 10 100 1000 10000
#     do
#             # cross validation
#         echo "running corss validation"
#         python testbench.py --use_cross_validation 1 \
#         --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat \
#         --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models/libSVM_c${c2}_b0.md \
#         --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_b0_weightM_c${c1}_mergeM_c${c2} \
#         --data_path ${input_folder}/test_bert.pickle \
#         --weight_model_ii_file ./meta_features/tdt4/tfidf_time.ii
#     done 
# done

for c1 in 1000
do
    # for c2 in 0.6 0.7 0.8 0.9 1.0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 10 100 1000 10000
    for c2 in 0.001 0.005 0.01 0.05 0.1 0.5 1.0 10 100 1000
    do
        # cross validation
        echo "running corss validation"
        python testbench.py --use_cross_validation 1 \
        --weight_model_dir ${input_folder}/${feature_option}/svm_triplet_weight_models/SVM_triplet_c${c1}.dat \
        --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
        --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_b0_weightM_c${c1}_mergeM_c${c2} \
        --data_path ${input_folder}/test_bert.pickle \
        --weight_model_ii_file ./meta_features/tdt4/tfidf_time.ii
    done 
done

        # --sklearn_model_specs tfidf_time-tdt4



feature_option=tfidf_time_esbert
input_folder=../output/exp_sbert_tdt4_ep5_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c10.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models/libSVM_c0.0001_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c10_mergeM_c0.0001 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/tdt4/tfidf_time.ii

python evaluate_model_outputs.py --dataset_name tdt4 --prediction_path ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c10_mergeM_c0.0001eng.out


#####################################################################
### TFIDF : TDT4 -- online training (SMOTE)
#####################################################################
feature_option=tfidf
# choose a different epoch ep5
for input_folder in ../output/exp_sbert_tdt4_ep5_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
do
    # python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

    # # train svm_rank models
    # for c1 in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10 100
    # do
    #     ## training SVM-rank Models
    #     mkdir ${input_folder}/${feature_option}/svm_rank_models
    #     ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ${input_folder}/${feature_option}/train_svm_rank.dat ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat
    # done

    # train libSVM merge models
    # python train_svm_merge.py --input_folder ${input_folder}/${feature_option}/ --use_smote 0 --features ${feature_option}
    # python train_svm_merge.py --input_folder ${input_folder}/${feature_option}/ --use_smote 1 --features ${feature_option}

    mkdir ${input_folder}/${feature_option}/predictions
    mkdir ${input_folder}/${feature_option}/predictions/cross_validations
    # #  0.00005 0.0001 0.0005 0.001 0.1 0.01 0.001
    for c1 in 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10 100
    do
        # for c2 in 0.6 0.7 0.8 0.9 1.0
        for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100
        do
            # cross validation
            echo "running corss validation"
            python testbench.py --use_cross_validation 1 \
            --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat \
            --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
            --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_b0_weightM_c${c1}_mergeM_c${c2} \
            --data_path ${input_folder}/test_bert.pickle \
            --weight_model_ii_file ./meta_features/tdt4/${feature_option}.ii
        done 
    done
done

feature_option=tfidf
# for input_folder in ../output/exp_sbert_tdt4_ep5_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
# do
#     # test ( b0 )
#     python testbench.py --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c0.1.dat \
#     --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c0.6_b0.md \
#     --output_filename ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c0.1_mergeM_c0.6 \
#     --data_path ${input_folder}/test_bert.pickle \
#     --weight_model_ii_file ./meta_features/tdt4/${feature_option}.ii

#     python evaluate_model_outputs.py --dataset_name tdt4 --prediction_path ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c0.1_mergeM_c0.6eng.out
# done

for input_folder in ../output/exp_sbert_tdt4_ep5_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
do
    # test ( b0 )
    python testbench.py --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c0.1.dat \
    --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c0.3_b0.md \
    --output_filename ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c0.1_mergeM_c0.3 \
    --data_path ${input_folder}/test_bert.pickle \
    --weight_model_ii_file ./meta_features/tdt4/${feature_option}.ii

    python evaluate_model_outputs.py --dataset_name tdt4 --prediction_path ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c0.1_mergeM_c0.3eng.out
done

# DO the following
# 1. modify `generate_svm_data.py`
# 2. modify `train_svm_merge.py`
# 3. modify the model path
#####################################################################
### TFIDF + TIME + SBERT : TDT4 -- online training (SMOTE)
#####################################################################
feature_option=tfidf_time_sbert
# use ep5 because this is rarely the best fine-tuned model
for input_folder in ../output/exp_sbert_tdt4_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
do
    # python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

    # # train svm_rank models
    # for c1 in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10 100
    # do
    #     ## training SVM-rank Models
    #     mkdir ${input_folder}/${feature_option}/svm_rank_models
    #     ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ${input_folder}/${feature_option}/train_svm_rank.dat ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat
    # done

    # train libSVM merge models
    # python train_svm_merge.py --input_folder ${input_folder}/${feature_option}/ --use_smote 0 --features ${feature_option}
    # python train_svm_merge.py --input_folder ${input_folder}/${feature_option}/ --use_smote 1 --features ${feature_option}

    mkdir ${input_folder}/${feature_option}/predictions
    mkdir ${input_folder}/${feature_option}/predictions/cross_validations
    # #  0.00005 0.0001 0.0005 0.001 0.1 0.01 0.001
    for c1 in 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10 100
    do
        # for c2 in 0.6 0.7 0.8 0.9 1.0
        for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
        do
            # cross validation
            echo "running corss validation"
            python testbench.py --use_cross_validation 1 \
            --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat \
            --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
            --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_b0_weightM_c${c1}_mergeM_c${c2} \
            --data_path ${input_folder}/test_bert.pickle \
            --weight_model_ii_file ./meta_features/tdt4/${feature_option}.ii
        done 
    done
done

feature_option=tfidf_time_sbert
for input_folder in ../output/exp_sbert_tdt4_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
do
    # test ( b0 )
    python testbench.py --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c10.dat \
    --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c0.15_b0.md \
    --output_filename ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c10_mergeM_c0.15 \
    --data_path ${input_folder}/test_bert.pickle \
    --weight_model_ii_file ./meta_features/tdt4/${feature_option}.ii

    python evaluate_model_outputs.py --dataset_name tdt4 --prediction_path ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c10_mergeM_c0.15eng.out
done

#####################################################################
### TFIDF + TIME + E-SBERT : TDT4 -- online training (SMOTE)
#####################################################################
feature_option=tfidf_time_esbert
# use ep5 because this is rarely the best fine-tuned model
for input_folder in ../output/exp_esbert_tdt4_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
do
    # python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

    # # train svm_rank models
    # for c1 in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10 100
    # do
    #     ## training SVM-rank Models
    #     mkdir ${input_folder}/${feature_option}/svm_rank_models
    #     ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ${input_folder}/${feature_option}/train_svm_rank.dat ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat
    # done

    # # train libSVM merge models
    # python train_svm_merge.py --input_folder ${input_folder}/${feature_option}/ --use_smote 0 --features ${feature_option}
    # python train_svm_merge.py --input_folder ${input_folder}/${feature_option}/ --use_smote 1 --features ${feature_option}

    mkdir ${input_folder}/${feature_option}/predictions
    mkdir ${input_folder}/${feature_option}/predictions/cross_validations
    # #  0.00005 0.0001 0.0005 0.001 0.1 0.01 0.001
    for c1 in 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10 100
    do
        # for c2 in 0.6 0.7 0.8 0.9 1.0
        for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000 100000 1000000
        do
            # cross validation
            echo "running corss validation"
            python testbench.py --use_cross_validation 1 \
            --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat \
            --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
            --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_b0_weightM_c${c1}_mergeM_c${c2} \
            --data_path ${input_folder}/test_bert.pickle \
            --weight_model_ii_file ./meta_features/tdt4/${feature_option}.ii
        done 
    done
done

feature_option=tfidf_time_esbert
for input_folder in ../output/exp_esbert_tdt4_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
do
    # test ( b0 )
    python testbench.py --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c10.dat \
    --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c0.15_b0.md \
    --output_filename ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c10_mergeM_c0.15 \
    --data_path ${input_folder}/test_bert.pickle \
    --weight_model_ii_file ./meta_features/tdt4/${feature_option}.ii

    python evaluate_model_outputs.py --dataset_name tdt4 --prediction_path ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c10_mergeM_c0.15eng.out
done



#####################################################################
### TFIDF + TIME + SinPE-E-SBERT : TDT4 -- online training (SMOTE)
#####################################################################
feature_option=tfidf_time_sinpe_esbert
model_folder=../output/exp_pos2vec_esbert_tdt4_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_month
# use ep5 because this is rarely the best fine-tuned model
for input_folder in ${model_folder}
do
    # python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

    # # train svm_rank models
    # for c1 in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10 100
    # do
    #     ## training SVM-rank Models
    #     mkdir ${input_folder}/${feature_option}/svm_rank_models
    #     ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ${input_folder}/${feature_option}/train_svm_rank.dat ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat
    # done

    # # train libSVM merge models
    # python train_svm_merge.py --input_folder ${input_folder}/${feature_option}/ --use_smote 0 --features ${feature_option}
    # python train_svm_merge.py --input_folder ${input_folder}/${feature_option}/ --use_smote 1 --features ${feature_option}

    mkdir ${input_folder}/${feature_option}/predictions
    mkdir ${input_folder}/${feature_option}/predictions/cross_validations
    # #  0.00005 0.0001 0.0005 0.001 0.1 0.01 0.001
    for c1 in 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10 100
    do
        # for c2 in 0.6 0.7 0.8 0.9 1.0
        for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000 100000 1000000
        do
            # cross validation
            echo "running corss validation"
            python testbench.py --use_cross_validation 1 \
            --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat \
            --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
            --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_b0_weightM_c${c1}_mergeM_c${c2} \
            --data_path ${input_folder}/test_bert.pickle \
            --weight_model_ii_file ./meta_features/tdt4/${feature_option}.ii
        done 
    done
done

feature_option=tfidf_time_sinpe_esbert
model_folder=../output/exp_pos2vec_esbert_tdt4_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_month
for input_folder in ${model_folder}
do
    # test ( b0 )
    python testbench.py --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c10.dat \
    --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c0.15_b0.md \
    --output_filename ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c10_mergeM_c0.15 \
    --data_path ${input_folder}/test_bert.pickle \
    --weight_model_ii_file ./meta_features/tdt4/${feature_option}.ii

    python evaluate_model_outputs.py --dataset_name tdt4 --prediction_path ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c10_mergeM_c0.15eng.out
done


#####################################################################
### TIME + SinPE-E-SBERT : TDT4 -- online training (SMOTE)
#####################################################################
feature_option=time_sinpe_esbert
model_folder=../output/exp_pos2vec_esbert_tdt4_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_month
# use ep5 because this is rarely the best fine-tuned model
for input_folder in ${model_folder}
do
    # python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

    # # train svm_rank models
    # for c1 in 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10 100
    # do
    #     ## training SVM-rank Models
    #     mkdir ${input_folder}/${feature_option}/svm_rank_models
    #     ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ${input_folder}/${feature_option}/train_svm_rank.dat ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat
    # done

    # # train libSVM merge models
    # python train_svm_merge.py --input_folder ${input_folder}/${feature_option}/ --use_smote 0 --features ${feature_option}
    # python train_svm_merge.py --input_folder ${input_folder}/${feature_option}/ --use_smote 1 --features ${feature_option}

    mkdir ${input_folder}/${feature_option}/predictions
    mkdir ${input_folder}/${feature_option}/predictions/cross_validations
    # #  0.00005 0.0001 0.0005 0.001 0.1 0.01 0.001
    for c1 in 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10 100
    do
        # for c2 in 0.6 0.7 0.8 0.9 1.0
        for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000 100000 1000000
        do
            # cross validation
            echo "running corss validation"
            python testbench.py --use_cross_validation 1 \
            --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat \
            --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
            --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_b0_weightM_c${c1}_mergeM_c${c2} \
            --data_path ${input_folder}/test_bert.pickle \
            --weight_model_ii_file ./meta_features/tdt4/${feature_option}.ii
        done 
    done
done

feature_option=time_sinpe_esbert
model_folder=../output/exp_pos2vec_esbert_tdt4_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_month
for input_folder in ${model_folder}
do
    # test ( b0 )
    python testbench.py --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c10.dat \
    --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c0.15_b0.md \
    --output_filename ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c10_mergeM_c0.15 \
    --data_path ${input_folder}/test_bert.pickle \
    --weight_model_ii_file ./meta_features/tdt4/${feature_option}.ii

    python evaluate_model_outputs.py --dataset_name tdt4 --prediction_path ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c10_mergeM_c0.15eng.out
done

#####################################################################
### SinPE-E-SBERT : TDT4 -- online training (SMOTE)
#####################################################################
feature_option=sinpe_esbert
model_folder=../output/exp_pos2vec_esbert_tdt4_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_month
# use ep5 because this is rarely the best fine-tuned model
for input_folder in ${model_folder}
do
    # python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

    # # train svm_rank models
    # for c1 in 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10 100
    # do
    #     ## training SVM-rank Models
    #     mkdir ${input_folder}/${feature_option}/svm_rank_models
    #     ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ${input_folder}/${feature_option}/train_svm_rank.dat ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat
    # done

    # # train libSVM merge models
    # python train_svm_merge.py --input_folder ${input_folder}/${feature_option}/ --use_smote 0 --features ${feature_option}
    # python train_svm_merge.py --input_folder ${input_folder}/${feature_option}/ --use_smote 1 --features ${feature_option}

    mkdir ${input_folder}/${feature_option}/predictions
    mkdir ${input_folder}/${feature_option}/predictions/cross_validations
    # #  0.00005 0.0001 0.0005 0.001 0.1 0.01 0.001
    for c1 in 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10 100
    do
        # for c2 in 0.6 0.7 0.8 0.9 1.0
        for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000 100000 1000000
        do
            # cross validation
            echo "running corss validation"
            python testbench.py --use_cross_validation 1 \
            --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat \
            --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
            --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_b0_weightM_c${c1}_mergeM_c${c2} \
            --data_path ${input_folder}/test_bert.pickle \
            --weight_model_ii_file ./meta_features/tdt4/${feature_option}.ii
        done 
    done
done

feature_option=sinpe_esbert
model_folder=../output/exp_pos2vec_esbert_tdt4_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss_time_month
for input_folder in ${model_folder}
do
    # test ( b0 )
    python testbench.py --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c10.dat \
    --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c0.15_b0.md \
    --output_filename ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c10_mergeM_c0.15 \
    --data_path ${input_folder}/test_bert.pickle \
    --weight_model_ii_file ./meta_features/tdt4/${feature_option}.ii

    python evaluate_model_outputs.py --dataset_name tdt4 --prediction_path ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c10_mergeM_c0.15eng.out
donenv