export CUDA_VISIBLE_DEVICES=1
#####################################################################
### TFIDF + TIME : TDT1 -- online training
#####################################################################
feature_option=tfidf_time
# use ep4 because this is rarely the best fine-tuned model
input_folder=../output/exp_sbert_tdt1_ep4_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
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
        --weight_model_ii_file ./meta_features/tfidf_time.ii

    done 
done

feature_option=tfidf_time
input_folder=../output/exp_sbert_tdt1_ep4_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c0.5.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models/libSVM_c10000_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c0.5_mergeM_c10000 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/tfidf_time.ii

python evaluate_model_outputs.py --dataset_name tdt1 --prediction_path ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c0.5_mergeM_c10000eng.out

#####################################################################
### TFIDF + TIME : TDT1 -- online training (SMOTE)
#####################################################################
feature_option=tfidf_time
# choose a different epoch ep5
input_folder=../output/exp_sbert_tdt1_ep5_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
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
        --merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c${c2}_b0.md \
        --output_filename ${input_folder}/${feature_option}/predictions/cross_validations/pred_b0_weightM_c${c1}_mergeM_c${c2} \
        --data_path ${input_folder}/test_bert.pickle \
        --weight_model_ii_file ./meta_features/tfidf_time.ii

    done 
done

feature_option=tfidf_time
input_folder=../output/exp_sbert_tdt1_ep5_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
# test ( b0 )
python testbench.py --weight_model_dir ${input_folder}/${feature_option}/svm_rank_models/weight_model_svmrank_c0.1.dat \
--merge_model_dir ${input_folder}/${feature_option}/svm_merge_models_smote/libSVM_c0.001_b0.md \
--output_filename ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c0.1_mergeM_c0.001 \
--data_path ${input_folder}/test_bert.pickle \
--weight_model_ii_file ./meta_features/tfidf_time.ii

python evaluate_model_outputs.py --dataset_name tdt1 --prediction_path ${input_folder}/${feature_option}/predictions/pred_b0_weightM_c0.1_mergeM_c0.001eng.out

