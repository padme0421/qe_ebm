# preprocess
#paste train_corpus1000.ko train_corpus1000.en | awk -F '\t' '{print $1 " ||| " $2}' > train_corpus1000.ko-en
paste sample_new_corpus.ko sample_new_corpus.en | awk -F '\t' '{print $1 " ||| " $2}' > sample_new_corpus.ko-en
