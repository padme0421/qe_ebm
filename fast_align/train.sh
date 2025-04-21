# source to target
build/fast_align -i train_corpus1000.ko-en -d -v -o -p fwd_params >fwd_align 2>fwd_err

# target to source
build/fast_align -i train_corpus1000.ko-en -r -d -v -o -p rev_params >rev_align 2>rev_err

# symmetrize
build/atools -i fwd_align -j rev_align -c grow-diag-final-and >sym_align