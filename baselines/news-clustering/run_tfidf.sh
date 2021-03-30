# for c1 in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5
# do
# 	for c2 in 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0
# 		do
# 			# ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ./svm_en_data/train_bert_rank_without_bert.dat ./svm_en_data/weight_models/model_tfidf_c${c1}.dat
# 			python testbench.py --weight_model_dir ./svm_en_data/weight_models/model_tfidf_c${c1}.dat --merge_model_dir ./svm_en_data/merge_models/tfidf/merge_model_tfidf_c${c2}.md --output_filename ./svm_en_data/output/tfidf_less_than_1/tfidf_weightM_c${c1}_mergeM_c${c2}
# 			# python eval.py clustering.eng.out dataset/dataset.test.json -f  
# 		done
# done

# SVM-rank + SVM-lib
for c1 in 0.0001 0.001
do
	for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1
	do
		python testbench.py --weight_model_dir ./svm_en_data/weight_models/svmrank_tfidf/model_c${c1}.dat \
		--merge_model_dir ./svm_en_data/merge_models/tfidf/merge_model_c${c2}.md \
		--output_filename ./svm_en_data/output/svm_tfidf/weightM_c${c1}_mergeM_c${c2}
	done
done

# SVM-rank + SVM-lib + oversampling
for c1 in 0.0001 0.001
do
	for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1
	do
		python testbench.py --weight_model_dir ./svm_en_data/weight_models/svmrank_tfidf/model_c${c1}.dat \
		--merge_model_dir ./svm_en_data/merge_models/tfidf_smote/merge_model_c${c2}.md \
		--output_filename ./svm_en_data/output/svm_tfidf_smote/weightM_c${c1}_mergeM_c${c2}
	done
done