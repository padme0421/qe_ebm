import pandas as pd
import argparse

def bleu_score_cor_bleu_cutoff(file):
    # read in a csv file containing bleu and score
    # cut off by bleu and see the correlation
    df = pd.read_csv(file)
    
    # first count bleu == 0 rows
    bleu_cutoff_df = df.loc[df['bleu'] > 0]
    print("number of rows with bleu == 0:", len(df) - len(bleu_cutoff_df))

    cor_matrix = bleu_cutoff_df.corr(method='pearson')
    correlation = cor_matrix['bleu']['score']
    return correlation


def bleu_score_cor_score_cutoff(file):
    df = pd.read_csv(file)
    df['quantile'] = pd.qcut(df['score'], 2, labels=False)
    score_cutoff_df = df.loc[df['quantile'] == 1] # higher half
    cor_matrix = score_cutoff_df.corr(method='pearson')
    correlation = cor_matrix['bleu']['score']
    return correlation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str)
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    if args.function == 'bleu_score_cor_bleu_cutoff':
        cor = bleu_score_cor_bleu_cutoff(args.file)
        print(cor)
    elif args.function == 'bleu_score_cor_score_cutoff':
        cor = bleu_score_cor_score_cutoff(args.file)
        print(cor)