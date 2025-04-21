#build/force_align.py fwd_params fwd_err rev_params rev_err grow-diag-final-and <sample_new_corpus.ko-en >sample_new_alignments.ko-en.gdfa

#while getopts fp:fe:rp:re:cp:al flag
#do
#    case "${flag}" in
#        fp) forward_params=${OPTARG};;
#        fe) forward_errors=${OPTARG};;
#        rp) reverse_params=${OPTARG};;
#        re) reverse_errors=${OPTARG};;
#        #cp) corpus=${OPTARG};;
#        #al) alignment_file=${OPTARG};;
#    esac
#done

#build/force_align.py ${fwd_params} ${pwd_err} ${rev_params} ${rev_err} grow-diag-final-and #<${corpus} >${alignment_file}
build/force_align.py fwd_params fwd_err rev_params rev_err grow-diag-final-and