import pandas as pd
import argparse

def make_word_translation_table(lexicon_file, output_file, top_k = 3):
    """
    for all words in corpus, make a csv file of the top k best translations from fast align
    """
    with open(lexicon_file, 'r') as f:
        lexicon = f.readlines()
    
    translation_table = dict()
    for line in lexicon:
        src_word, trg_word, prob= line.split()
        prob = float(prob)
        if src_word in translation_table:
            trg_list = translation_table[src_word]
            trg_list.append((trg_word, prob))
            trg_list.sort(key=lambda tup: tup[1], reverse=True) # sort by prob
            if len(trg_list) > top_k:
                trg_list = trg_list[:top_k] # cut off
            translation_table[src_word] = trg_list
        else:
            translation_table[src_word] = [(trg_word, prob)]

    translation_table_df = pd.DataFrame(columns=["src", "trg", "prob"])
    for src_word in translation_table:
        trg_list = translation_table[src_word]
        for (trg_word, trg_prob) in trg_list:
            translation_table_df.loc[len(translation_table_df.index)] = [src_word, trg_word, trg_prob]
    
    translation_table_df.to_csv(output_file)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexicon_file", "-l", default="fwd_params")
    parser.add_argument("--output_file", "-o", default="lexicon.csv")
    parser.add_argument("--top_k", "-k", default=3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    make_word_translation_table(args.lexicon_file, args.output_file, 3)