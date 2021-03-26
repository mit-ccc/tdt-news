# 0.0001 0.001 0.01 
# for c1 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1
# do
# for c2 in 0.005 0.01 0.05 0.1 0.5 1.0
# do
# # ./svm_rank/svm_rank_learn -c ${c} -t 0 -d 3 -g 1 -s 1 -r 1 ./svm_en_data/train_bert_rank.dat ./svm_en_data/model_c${c}.dat
# # python testbench.py --weight_model_dir ./models/en/4_1491902620.876421_10000.0.model --merge_model_dir ./svm_en_data/merge_models/tfidf_smote/merge_model_c${c2}.md --output_filename ./svm_en_data/output/tfidf_smote/weightM_marinda_mergeM_c${c2}
# python testbench.py --weight_model_dir ./svm_en_data/weight_models/model_tfidf_c0.0001.dat --merge_model_dir ./svm_en_data/merge_models/tfidf_smote/merge_model_c${c2}.md --output_filename ./svm_en_data/output/tfidf_smote/weightM_c0.0001_mergeM_c${c2}
# done
# done


# ./svm_en_data/output/weightM_c0.001_mergeM_c0.0001eng.out
# precision: 0.8803148866634329; recall: 0.928470184356715; f-1: 0.9037515160951414
# for TF-IDF + BERT
# for c2 in 0.001 0.01 0.0001 0.0005 0.005 0.05 0.1 0.5 1.0 
# do
# # ./svm_rank/svm_rank_learn -c ${c} -t 0 -d 3 -g 1 -s 1 -r 1 ./svm_en_data/train_bert_rank.dat ./svm_en_data/model_c${c}.dat
# # python testbench.py --weight_model_dir ./models/en/4_1491902620.876421_10000.0.model --merge_model_dir ./svm_en_data/merge_models/tfidf_bert_smote/merge_model_c${c2}.md --output_filename ./svm_en_data/output/tfidf_smote/weightM_marinda_mergeM_c${c2}
# python testbench.py --weight_model_dir ./svm_en_data/weight_models/model_c0.001.dat --merge_model_dir ./svm_en_data/merge_models/tfidf_bert_smote/merge_model_c${c2}.md --output_filename ./svm_en_data/output/tfidf_bert_smote/weightM_c0.0001_mergeM_c${c2}
# done


# SVM-rank + SVM-lib + BERT 
for c1 in 0.001 0.0001
do
	for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1
	do
		python testbench.py --weight_model_dir ./svm_en_data/weight_models/svmrank_bert_tfidf/model_c${c1}.dat \
		--merge_model_dir ./svm_en_data/merge_models/tfidf_bert/merge_model_c${c2}.md \
		--output_filename ./svm_en_data/output/svm_tfidf_bert/weightM_c${c1}_mergeM_c${c2}
	done
done

# SVM-rank + SVM-lib + BERT + oversampling
for c1 in 0.001 0.0001
do
	for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1
	do
		python testbench.py --weight_model_dir ./svm_en_data/weight_models/svmrank_bert_tfidf/model_c${c1}.dat \
		--merge_model_dir ./svm_en_data/merge_models/tfidf_bert_smote/merge_model_c${c2}.md \
		--output_filename ./svm_en_data/output/svm_tfidf_bert_smote/weightM_c${c1}_mergeM_c${c2}
	done
done