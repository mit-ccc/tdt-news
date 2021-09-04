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
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c100_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c100_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c100_b0eng.out


# ==SUGGEST CONFIGURATION==
#                                            filename  precision     recall     fscore
# 7    pred_margin_ranking_lr0.1_libSVM_c10000_b0.csv  74.749510  79.454994  77.009649
# 33    pred_margin_ranking_lr0.1_libSVM_c1000_b0.csv  75.116982  79.010917  76.995318
# 35  pred_margin_ranking_lr0.01_libSVM_c10000_b0.csv  76.205350  77.810770  76.991922
# 34     pred_margin_ranking_lr0.1_libSVM_c100_b0.csv  74.430308  79.782219  76.984004
# 28    pred_margin_ranking_lr0.01_libSVM_c100_b0.csv  75.720756  78.277840  76.962817

feature_option=tfidf
input_folder=../output/exp_sbert_news2013_ep5_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
for lr in 0.1 0.5 1.0 10 100 1000
do
    for c2 in 10000
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

# ==SUGGEST CONFIGURATION==
#                                            filename  precision     recall     fscore
# 29   pred_margin_ranking_lr1.0_libSVM_c10000_b0.csv  74.841833  79.393144  77.042083
# 46   pred_margin_ranking_lr100_libSVM_c10000_b0.csv  75.463337  78.699246  77.040386
# 9    pred_margin_ranking_lr0.1_libSVM_c10000_b0.csv  74.749510  79.454994  77.009649
# 39    pred_margin_ranking_lr0.1_libSVM_c1000_b0.csv  75.116982  79.010917  76.995318
# 41  pred_margin_ranking_lr0.01_libSVM_c10000_b0.csv  76.205350  77.810770  76.991922

# apply the best configuration on the test set
feature_option=tfidf
input_folder=../output/exp_sbert_news2013_ep5_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr1.0_ep30.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c10000_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr1.0_libSVM_c10000_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr1.0_libSVM_c10000_b0eng.out

# ../output/exp_sbert_news2013_ep5_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/tfidf/predictions/pred_margin_ranking_lr1.0_libSVM_c10000_b0eng.out
# predicted cluster num: 761
# precision: 82.35; recall: 74.73; f-1: 78.36


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
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr0.001_ep30.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c0.1_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_b1_weightM_c0.001_mergeM_c0.1 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_b1_weightM_c0.001_mergeM_c0.1eng.out

# ==SUGGEST CONFIGURATION==
#                                            filename  precision     recall     fscore
# 31           pred_b1_weightM_c0.001_mergeM_c0.1.csv  92.359758  88.821419  90.555082
# 4    pred_margin_ranking_lr1.0_libSVM_c1e-05_b0.csv  95.816369  85.788868  90.522617
# 15    pred_margin_ranking_lr10_libSVM_c1e-05_b0.csv  89.781791  86.425423  88.066753
# 29  pred_margin_ranking_lr0.01_libSVM_c1e-05_b0.csv  89.234215  86.455437  87.811752
# 53   pred_margin_ranking_lr0.5_libSVM_c1e-05_b0.csv  88.006666  86.427374  87.204923

feature_option=tfidf_time
input_folder=../output/exp_sbert_news2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
for lr in 0.5 1.0 10 100 1000
do
    # for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
    for c2 in 1e-05 0.1
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

feature_option=tfidf_time
input_folder=../output/exp_sbert_news2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr1.0_ep30.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c1e-05_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr1.0_libSVM_c1e-05_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr1.0_libSVM_c1e-05_b0eng.out


hjian42@matlaber7:~$ screen -r 860077


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
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c1e-05_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c1e-05_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c1e-05_b0eng.out

# ==SUGGEST CONFIGURATION==
#                                             filename  precision     recall     fscore
# 8     pred_margin_ranking_lr0.5_libSVM_c1e-05_b0.csv  95.193343  84.663392  89.616519
# 11    pred_margin_ranking_lr0.1_libSVM_c1e-05_b0.csv  88.122203  85.086894  86.561425
# 5    pred_margin_ranking_lr0.01_libSVM_c1e-05_b0.csv  85.327206  85.152577  85.234120
# 4   pred_margin_ranking_lr0.001_libSVM_c1e-05_b0.csv  77.923677  87.236904  82.309203
# 2     pred_margin_ranking_lr0.5_libSVM_c5e-05_b0.csv  96.935713  69.078028  80.663561

feature_option=tfidf_time_sbert
input_folder=../output/exp_sbert_news2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
for lr in 1.0 10 100 1000
do
    # for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
    for c2 in 1e-05
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

feature_option=tfidf_time_sbert
input_folder=../output/exp_sbert_news2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
merge_folder=../output/exp_sbert_news2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/tfidf_time
for c2 in 1e-05 0.1
do
    # for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
    for lr in 0.001 0.01 0.1 0.5 1.0 10 100 1000
    do
        # cross validation
        echo "running corss validation"
        python testbench.py --use_cross_validation 1 \
        --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr${lr}_ep30.dat\
        --merge_model_dir ${merge_folder}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
        --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_margin_ranking_lr${lr}_libSVM_tfidftime_c${c2}_b0 \
        --data_path ${input_folder}/test_bert.pickle \
        --weight_model_ii_file ./meta_features/${feature_option}.ii
    done 
done

# apply the best configuration on the test set
feature_option=tfidf_time_sbert
input_folder=../output/exp_sbert_news2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr0.5_ep30.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c1e-05_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.5_libSVM_tfidftime_c1e-05_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.5_libSVM_tfidftime_c1e-05_b0eng.out

# testing en #docs 8726
# ../output/exp_sbert_news2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/tfidf_time_sbert/predictions/pred_margin_ranking_lr0.5_libSVM_tfidftime_c1e-05_b0eng.out
# predicted cluster num: 767
# precision: 96.71; recall: 80.41; f-1: 87.81

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
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c1e-05_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c1e-05_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c1e-05_b0eng.out

# ==SUGGEST CONFIGURATION==
#                                              filename  precision     recall     fscore
# 1   pred_margin_ranking_lr1.0_libSVM_tfidftime_c1e...  95.805351  85.019482  90.083023
# 14  pred_margin_ranking_lr0.5_libSVM_tfidftime_c1e...  94.587150  85.033180  89.546730
# 3      pred_margin_ranking_lr1.0_libSVM_c1e-05_b0.csv  95.878863  83.423061  89.212384
# 23     pred_margin_ranking_lr0.5_libSVM_c1e-05_b0.csv  94.741655  83.536596  88.777841
# 18  pred_margin_ranking_lr0.1_libSVM_tfidftime_c1e...  91.866936  84.268839  87.898013

feature_option=tfidf_time_esbert
input_folder=../output/exp_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
for lr in 1.0 10 100 1000
do
    # for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
    for c2 in 0.0001
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


feature_option=tfidf_time_esbert
input_folder=../output/exp_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
merge_folder=../output/exp_sbert_news2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/tfidf_time
for c2 in 1e-05 0.1
do
    # for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
    for lr in 0.001 0.01 0.1 0.5 1.0 10 100 1000
    do
        # cross validation
        echo "running corss validation"
        python testbench.py --use_cross_validation 1 \
        --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr${lr}_ep30.dat\
        --merge_model_dir ${merge_folder}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
        --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_margin_ranking_lr${lr}_libSVM_tfidftime_c${c2}_b0 \
        --data_path ${input_folder}/test_bert.pickle \
        --weight_model_ii_file ./meta_features/${feature_option}.ii
    done 
done

merge_folder=../output/exp_sbert_news2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/tfidf_time
# apply the best configuration on the test set
feature_option=tfidf_time_esbert
input_folder=../output/exp_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr1.0_ep30.dat \
--merge_model_dir ${merge_folder}/svm_merge_models_smote/libSVM_c1e-05_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr1.0_libSVM_c1e-05_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr1.0_libSVM_c1e-05_b0eng.out



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
input_folder=../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr0.01_ep30.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c1e-05_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c1e-05_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c1e-05_b0eng.out

# ../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/tfidf_time_sinpe_esbert/predictions/pred_margin_ranking_lr0.01_libSVM_c1e-05_b0eng.out
# predicted cluster num: 714
# precision: 95.21; recall: 84.30; f-1: 89.43

#                                            filename  precision     recall     fscore
# 7    pred_margin_ranking_lr0.1_libSVM_c1e-05_b0.csv  92.074275  84.375588  88.053612
# 3   pred_margin_ranking_lr0.01_libSVM_c1e-05_b0.csv  88.149082  84.953368  86.515912
# 2  pred_margin_ranking_lr0.001_libSVM_c1e-05_b0.csv  89.010565  79.594956  84.034095
# 4    pred_margin_ranking_lr0.1_libSVM_c5e-05_b0.csv  94.842742  67.662913  78.975778
# 0   pred_margin_ranking_lr0.01_libSVM_c5e-05_b0.csv  90.086275  70.158599  78.873176

feature_option=tfidf_time_sinpe_esbert
input_folder=../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
for lr in 0.5 1.0 10 100 1000
do
    # for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
    for c2 in 1e-05 10000
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

feature_option=tfidf_time_sinpe_esbert
input_folder=../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
merge_folder=../output/exp_sbert_news2013_ep1_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/tfidf_time
for c2 in 1e-05 0.1
do
    # for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0 10 100 1000 10000
    for lr in 0.001 0.01 0.1 0.5 1.0 10 100 1000
    do
        # cross validation
        echo "running corss validation"
        python testbench.py --use_cross_validation 1 \
        --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr${lr}_ep30.dat\
        --merge_model_dir ${merge_folder}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
        --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_margin_ranking_lr${lr}_libSVM_tfidftime_c${c2}_b0 \
        --data_path ${input_folder}/test_bert.pickle \
        --weight_model_ii_file ./meta_features/${feature_option}.ii
    done 
done


# apply the best configuration on the test set
feature_option=tfidf_time_sinpe_esbert
input_folder=../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr100_ep30.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c1e-05_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr100_libSVM_c1e-05_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr100_libSVM_c1e-05_b0eng.out


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
input_folder=../output/exp_pos2vec_esbert_news2013_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/margin_ranking_weight_models/margin_ranking_lr0.01_ep30.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c10000_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c10000_b0 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/${feature_option}.ii

# decide "pred_b0_weightM_c0.5_mergeM_c10000" is the best configuration
# and put in the right models using "pred_b0_weightM_c0.5_mergeM_c10000" --> weight_model_svmrank_c0.5.dat and libSVM_c10000_b0.md
python evaluate_model_outputs.py --dataset_name news2013 --prediction_path ${input_folder}/${feature_option}/predictions/pred_margin_ranking_lr0.01_libSVM_c10000_b0eng.out

