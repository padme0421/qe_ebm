import argparse

def align_id_to_word(alignment_file, src_corpus_file, trg_corpus_file): 
    """
    for easier visual inspection,
    change id alignments formatted as (0-1, 1-2, ...) to word alignments in string format ("boy"-"kinder")
    """

    # read files
    #alignment_file = 'sym_align'
    with open(alignment_file, 'r') as f:
        id_alignments = f.readlines()
    
    #src_corpus_file = 'train_corpus1000.ko'
    with open(src_corpus_file, 'r') as f:
        src_corpus = f.readlines()

    #trg_corpus_file = 'train_corpus1000.en'
    with open(trg_corpus_file, 'r') as f:
        trg_corpus = f.readlines()
    
    # change id alignment to word alignments
    word_alignments = []
    for (src, trg, id_alignment) in zip(src_corpus, trg_corpus, id_alignments):
        src_word_list = src.split()
        trg_word_list = trg.split()
        id_pairs = id_alignment.split()
        word_alignment = []
        for id_pair in id_pairs:
            src_id, trg_id = int(id_pair.split("-")[0]), int(id_pair.split("-")[1])
            word_pair = (src_word_list[src_id], trg_word_list[trg_id]) # "boy", "kinder"
            word_alignment.append(word_pair) # [(), ()]
        word_alignments.append(word_alignment) # [[(), ()], [(), ()]]

    # write word alignments to file
    new_alignment_file = alignment_file + "_word"
    with open(new_alignment_file, 'w') as f:
        for word_alignment in word_alignments:
            word_pair_str_list = []
            for word_pair in word_alignment:
                word_pair_str = "-".join(word_pair) # "boy"-"kinder"
                word_pair_str_list.append(word_pair_str)
            word_alignment_str = " ".join(word_pair_str_list)
            f.write(word_alignment_str + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment_file", "-a", default="sym_align")
    parser.add_argument("--src_corpus_file", "-s", default="train_corpus1000.ko")
    parser.add_argument("--trg_corpus_file", "-t", default="train_corpus1000.en")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    align_id_to_word(args.alignment_file, args.src_corpus_file, args.trg_corpus_file)