export CUDA_VISIBLE_DEVICES=1
#####################################################################
### TFIDF + TIME : News2013 -- Miranda et al. (2018) baseline############
######### SVM-rank weight model + SVM-lib cluster creation model#####
#####################################################################
feature_option=tfidf_time
# use ep4 because this is rarely the best fine-tuned model
input_folder=../output/exp_sbert_new2013_ep4_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

# train svm_rank models
for c1 in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 10 100 1000 10000
do
    mkdir ${input_folder}/${feature_option}/svm_rank_models
    ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ${input_folder}/${feature_option}/train_svm_rank.dat ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat
done

# train libSVM merge models
python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_balanced.dat --use_smote 0 --features ${feature_option}
python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_raw.dat --use_smote 1 --features ${feature_option}
# python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_raw.dat --use_smote 0 --features ${feature_option}
# python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_raw.dat --use_smote 1 --features ${feature_option}

# run cross validation
mkdir ${input_folder}/${feature_option}/predictions
mkdir ${input_folder}/${feature_option}/predictions/cross_validations
#  0.00005 0.0001 0.0005 0.001 0.1 0.01 0.001
for c1 in 0.1 0.01 0.001 0.0001 0.00001 0.5 1.0 10
do
    # for c2 in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    for c2 in 0.0001 0.001 0.005 0.01 0.1 0.5 1.0 10 100 1000 10000
    do
            # cross validation
        echo "running corss validation"
        python testbench.py --use_cross_validation 1 \
        --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c${c1}.dat \
        --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models/libSVM_c${c2}_b0.md \
        --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_b0_weightM_c${c1}_mergeM_c${c2} \
        --data_path ${input_folder}/test_bert.pickle \
        --weight_model_ii_file ./meta_features/${feature_option}.ii

    done 
done

# apply the best configuration on the test set
feature_option=tfidf_time
input_folder=../output/exp_sbert_new2013_ep4_mgn2.0_btch32_norm1.0_max_seq_230_sample_random

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c0.5.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models/libSVM_c10000_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c0.5_mergeM_c10000 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

python evaluate_model_outputs.py --dataset_name new2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c0.5_mergeM_c10000eng.out

###########################################################################
### TFIDF + TIME + ESBERT : News2013 -- Saravanakumar et al. (2021) baseline###
######### SVM-tripet weight model + NN-LBFGS cluster creation model########
###########################################################################
feature_option=tfidf_time_esbert
# use the best esbert model
input_folder=../output/exp_esbert_new2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

# train SVM-triplet weight models
python train_weight_model_svm_triplet.py --input_folder ${input_folder}/${feature_option}

# generate data for mege model with the help of one picked weighting model
# for instance, I pick `SVM_triplet_c1.0.dat`, you can try a few such as [0.001 0.01 0.1 0.5 1.0 10 100]
weight_model_c=1000
python generate_lbfgs_data.py --input_folder ${input_folder} \
--features ${feature_option} \
--weight_model ${input_folder}/${feature_option}/svm_triplet_weight_models/SVM_triplet_c${weight_model_c}.dat \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# train NN-LBFGS merge models using the data we just generated `train_lbfgs_raw_c1.0.dat`
# note that the suffix is c1.0 because we want to use the data generated with `SVM_triplet_c1.0.dat`
python train_merge_model_nn_lbfgs.py --data_path ${input_folder}/${feature_option}/train_lbfgs_raw_c${weight_model_c}.dat

# run cross validation
mkdir ${input_folder}/${feature_option}/predictions
mkdir ${input_folder}/${feature_option}/predictions/cross_validations
# pick c1 that is used earlier
for c1 in ${weight_model_c}
do
    for c2 in 0.0001 0.001 0.01 0.1 0.5 1.0 10 100 1000
    do
            # cross validation
        echo "running corss validation"
        python testbench.py --use_cross_validation 1 \
        --weight_model_dir ${input_folder}/${feature_option}/svm_triplet_weight_models/SVM_triplet_c${c1}.dat \
        --merge_model_dir ${input_folder}/${feature_option}/nn_lbfgs_merge_models/nn_lbfgs_merge_lr${c2}.md.pkl \
        --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_weightM_c${c1}_mergeM_c${c2} \
        --data_path ${input_folder}/test_bert.pickle \
        --weight_model_ii_file ./meta_features/${feature_option}.ii \
        --sklearn_model_specs ${feature_option}-new2013
    done 
done


# apply the best configuration on the test set
feature_option=tfidf_time_esbert
input_folder=../output/exp_esbert_new2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random

# # decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# # e.g. put in the right models using "pred_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python testbench.py --use_cross_validation 1 \
--weight_model_dir ${input_folder}/${feature_option}/svm_triplet_weight_models/SVM_triplet_c1000.dat \
--merge_model_dir ${input_folder}/${feature_option}/nn_lbfgs_merge_models/nn_lbfgs_merge_lr0.1.md \
--output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_weightM_c1000_mergeM_c0.1 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii
python evaluate_model_outputs.py --dataset_name new2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_weightM_c1000_mergeM_c0.1eng.out

#########################################################################################
#########################################################################################
######### Ours: LR-MarginRanking weight model + SVM-lib-SMOTE cluster creation model#####
################ use MAX-SIM for cluster creation  model ################################
#########################################################################################


################################### TFIDF ###############################################
feature_option=tfidf
# use ep4 because this is rarely the best fine-tuned model
input_folder=../output/exp_sbert_news2013_ep5_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
# python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

# train LR_margin_ranking weight models
python train_weight_model_margin_ranking.py --input_folder ${input_folder}/${feature_option}/

# train libSVM merge models
# python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_balanced.dat --use_smote 0 --features ${feature_option}
python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_raw.dat --use_smote 1 --features ${feature_option}

# run cross validation
mkdir ${input_folder}/${feature_option}/predictions
mkdir ${input_folder}/${feature_option}/predictions/cross_validations
# 0.1 0.01 0.001 0.0001 0.00001 0.5 1.0 10
for lr in 0.001 0.01 0.1 0.5 1.0 10 100 1000
do
    for c2 in 1e-05 5e-05 0.0001 0.001 0.01 0.1 0.5 1.0 10 100 1000 10000
    do
        # cross validation
        echo "running corss validation"
        python testbench.py --use_cross_validation 1 \
        --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr${lr}_ep30.dat\
        --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
        --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_margin_ranking_lr${lr}_libSVM_c${c2}_b0 \
        --data_path ${input_folder}/test_bert.pickle \
        --weight_model_ii_file ./meta_features/${feature_option}.ii
    done 
done

# apply the best configuration on the test set
feature_option=tfidf
input_folder=../output/exp_sbert_news2013_ep5_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr0.01_ep30.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c10000_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c10000_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c10000_b0eng.out

################################### TFIDF + TIME ###############################################
feature_option=tfidf_time
# use ep4 because this is rarely the best fine-tuned model
input_folder=../output/exp_sbert_news2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
# python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

# train LR_margin_ranking weight models
python train_weight_model_margin_ranking.py --input_folder ${input_folder}/${feature_option}/

# train libSVM merge models
# python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_balanced.dat --use_smote 0 --features ${feature_option}
python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_raw.dat --use_smote 1 --features ${feature_option}

# run cross validation
mkdir ${input_folder}/${feature_option}/predictions
mkdir ${input_folder}/${feature_option}/predictions/cross_validations
# 0.1 0.01 0.001 0.0001 0.00001 0.5 1.0 10
for lr in 0.001 0.01 0.1 0.5 1.0 10 100 1000
do
    # for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
    for c2 in 1e-05 5e-05 0.0001 0.001 0.01 0.1 0.5 1.0 10 100 1000 10000
    do
        # cross validation
        echo "running corss validation"
        python testbench.py --use_cross_validation 1 \
        --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr${lr}_ep30.dat\
        --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
        --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_margin_ranking_lr${lr}_libSVM_c${c2}_b0 \
        --data_path ${input_folder}/test_bert.pickle \
        --weight_model_ii_file ./meta_features/${feature_option}.ii
    done 
done

# apply the best configuration on the test set
feature_option=tfidf_time
input_folder=../output/exp_sbert_news2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr0.01_ep30.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c10000_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c10000_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c10000_b0eng.out

################################### TFIDF + TIME + E-SBERT ###############################################
feature_option=tfidf_time_esbert
# use ep4 because this is rarely the best fine-tuned model
input_folder=../output/exp_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

# train LR_margin_ranking weight models
python train_weight_model_margin_ranking.py --input_folder ${input_folder}/${feature_option}/

# train libSVM merge models
# python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_balanced.dat --use_smote 0 --features ${feature_option}
python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_raw.dat --use_smote 1 --features ${feature_option}

# run cross validation
mkdir ${input_folder}/${feature_option}/predictions
mkdir ${input_folder}/${feature_option}/predictions/cross_validations
# 0.1 0.01 0.001 0.0001 0.00001 0.5 1.0 10
for lr in 0.001 0.01 0.1 0.5 1.0 10 100 1000
do
    # for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
    for c2 in 1e-05 5e-05 0.0001 0.001 0.01 0.1 0.5 1.0 10 100 1000 10000
    do
        # cross validation
        echo "running corss validation"
        python testbench.py --use_cross_validation 1 \
        --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr${lr}_ep30.dat\
        --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
        --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_margin_ranking_lr${lr}_libSVM_c${c2}_b0 \
        --data_path ${input_folder}/test_bert.pickle \
        --weight_model_ii_file ./meta_features/${feature_option}.ii
    done 
done

# apply the best configuration on the test set
feature_option=tfidf_time_esbert
input_folder=../output/exp_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr0.01_ep30.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c10000_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c10000_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c10000_b0eng.out

################################### TFIDF + TIME + SBERT ###############################################
feature_option=tfidf_time_sbert
# use ep4 because this is rarely the best fine-tuned model
input_folder=../output/exp_sbert_news2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

# train LR_margin_ranking weight models
python train_weight_model_margin_ranking.py --input_folder ${input_folder}/${feature_option}/

# train libSVM merge models
# python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_balanced.dat --use_smote 0 --features ${feature_option}
python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_raw.dat --use_smote 1 --features ${feature_option}

# run cross validation
mkdir ${input_folder}/${feature_option}/predictions
mkdir ${input_folder}/${feature_option}/predictions/cross_validations
# 0.1 0.01 0.001 0.0001 0.00001 0.5 1.0 10
for lr in 0.001 0.01 0.1 0.5 1.0 10 100 1000
do
    # for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
    for c2 in 1e-05 5e-05 0.0001 0.001 0.01 0.1 0.5 1.0 10 100 1000 10000
    do
        # cross validation
        echo "running corss validation"
        python testbench.py --use_cross_validation 1 \
        --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr${lr}_ep30.dat\
        --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
        --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_margin_ranking_lr${lr}_libSVM_c${c2}_b0 \
        --data_path ${input_folder}/test_bert.pickle \
        --weight_model_ii_file ./meta_features/${feature_option}.ii
    done 
done

# apply the best configuration on the test set
feature_option=tfidf_time_sbert
input_folder=../output/exp_sbert_news2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr0.01_ep30.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c10000_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c10000_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c10000_b0eng.out


################################### TFIDF + TIME + sinpe_esbert ###############################################
feature_option=tfidf_time_sinpe_esbert
# use ep4 because this is rarely the best fine-tuned model
input_folder=../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

# train LR_margin_ranking weight models
python train_weight_model_margin_ranking.py --input_folder ${input_folder}/${feature_option}/

# train libSVM merge models
# python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_balanced.dat --use_smote 0 --features ${feature_option}
python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_raw.dat --use_smote 1 --features ${feature_option}

# run cross validation
mkdir ${input_folder}/${feature_option}/predictions
mkdir ${input_folder}/${feature_option}/predictions/cross_validations
# 0.1 0.01 0.001 0.0001 0.00001 0.5 1.0 10
for lr in 0.001 0.01 0.1 0.5 1.0 10 100 1000
do
    # for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
    for c2 in 1e-05 5e-05 0.0001 0.001 0.01 0.1 0.5 1.0 10 100 1000 10000
    do
        # cross validation
        echo "running corss validation"
        python testbench.py --use_cross_validation 1 \
        --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr${lr}_ep30.dat\
        --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
        --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_margin_ranking_lr${lr}_libSVM_c${c2}_b0 \
        --data_path ${input_folder}/test_bert.pickle \
        --weight_model_ii_file ./meta_features/${feature_option}.ii
    done 
done

# apply the best configuration on the test set
feature_option=tfidf_time_sinpe_esbert
model_folder=../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr0.01_ep30.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c10000_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c10000_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c10000_b0eng.out

################################### TFIDF  + sinpe_esbert ###############################################
feature_option=tfidf_sinpe_esbert
# use ep4 because this is rarely the best fine-tuned model
input_folder=../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
python generate_svm_data.py --input_folder ${input_folder} --features ${feature_option}

# train LR_margin_ranking weight models
python train_weight_model_margin_ranking.py --input_folder ${input_folder}/${feature_option}/

# train libSVM merge models
# python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_balanced.dat --use_smote 0 --features ${feature_option}
python train_merge_model_svm.py --data_path ${input_folder}/${feature_option}/train_svmlib_raw.dat --use_smote 1 --features ${feature_option}

# run cross validation
mkdir ${input_folder}/${feature_option}/predictions
mkdir ${input_folder}/${feature_option}/predictions/cross_validations
# 0.1 0.01 0.001 0.0001 0.00001 0.5 1.0 10
for lr in 0.001 0.01 0.1 0.5 1.0 10 100 1000
do
    # for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
    for c2 in 1e-05 5e-05 0.0001 0.001 0.01 0.1 0.5 1.0 10 100 1000 10000
    do
        # cross validation
        echo "running corss validation"
        python testbench.py --use_cross_validation 1 \
        --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr${lr}_ep30.dat\
        --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
        --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_margin_ranking_lr${lr}_libSVM_c${c2}_b0 \
        --data_path ${input_folder}/test_bert.pickle \
        --weight_model_ii_file ./meta_features/${feature_option}.ii
    done 
done

# apply the best configuration on the test set
feature_option=tfidf_sinpe_esbert
model_folder=../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr0.01_ep30.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c10000_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c10000_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c10000_b0eng.out

