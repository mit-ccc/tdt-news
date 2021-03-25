# 0.0001 0.001 0.01 
for c1 in 0.1 1.0 10 100 1000 10000 100000 1000000
do
    for c2 in 0.00005 0.0001 0.001 0.01 0.1 1.0 10 100 1000 10000 100000 1000000
    do
    # ./svm_rank/svm_rank_learn -c ${c} -t 0 -d 3 -g 1 -s 1 -r 1 ./svm_en_data/train_bert_rank.dat ./svm_en_data/model_c${c}.dat
    python testbench.py --weight_model_dir ./svm_en_data/weight_models/model_c${c1}.dat --merge_model_dir ./svm_en_data/merge_models/merge_model_c${c2}.md --output_filename ./svm_en_data/output/weightM_c${c1}_mergeM_c${c2}
    # python eval.py clustering.eng.out dataset/dataset.test.json -f  
    done
done
