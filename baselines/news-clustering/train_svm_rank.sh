# using only TFIDF features
for c1 in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 10 100
do
	# ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ./svm_en_data/train_bert_rank_without_bert.dat ./svm_en_data/weight_models/model_tfidf_c${c1}.dat
	./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ./svm_en_data/train_svm_rank_without_bert.dat ./svm_en_data/weight_models/svmrank_tfidf/model_c${c1}.dat
done

# using TFIDF + BERT features
for c1 in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 10 100
do
	# ./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ./svm_en_data/train_bert_rank.dat ./svm_en_data/weight_models/model_c${c1}.dat
	./svm_rank/svm_rank_learn -c ${c1} -t 0 -d 3 -g 1 -s 1 -r 1 ./svm_en_data/train_svm_rank.dat ./svm_en_data/weight_models/svmrank_bert_tfidf/model_c${c1}.dat
done