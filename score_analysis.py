import pandas as pd
import re
import numpy as np

def process_tensor_str(tensor_str):
    # temp
    #tensor([0.2330, 0.2257, 0.2968, 0.1186, 0.1648, 0.2842, 0.3419, 0.1357, 0.1297,
    #    0.1739, 0.0803, 0.3124, 0.1296, 0.1709, 0.1530, 0.1510, 0.1163, 0.1919,
    #    0.1243, 0.1711, 0.2576, 0.1676, 0.2439, 0.0939, 0.1718, 0.0923, 0.1405,
    #    0.2123, 0.1805, 0.0994, 0.0661, 0.1563, 0.3288, 0.1578, 0.1962, 0.0667,
    #    0.1731, 0.2088, 0.1695, 0.1537, 0.2724, 0.1968, 0.2431, 0.2125, 0.3310,
    #    0.1092, 0.2801, 0.1061, 0.1885, 0.1475, 0.1988, 0.1247, 0.0888, 0.2104,
    #    0.1503, 0.2696, 0.1342, 0.1459, 0.1532, 0.1533, 0.1030, 0.1051, 0.1243,
    #    0.1096])

    pattern = '[+-]?[0-9]+[.][0-9]+'
    scores = re.findall(pattern, tensor_str)
    scores = [float(score) for score in scores] 
    for score in scores:
        print("%f" % float(score))
    return scores

def main():
    # read csv file ("epoch_scores.csv")
    score_file = "2023-06-01-05:31:19-PM/epoch_scores.csv"
    df = pd.read_csv(score_file)
    epoch_mean_scores = []
    epoch_score_std = []
    # process tensors
    # get mean score per epoch
    columns = list(df)[1:] 
    print(columns)
    print(len(columns))
    for col in columns: # by epoch
        epoch_scores = []
        for row in range(len(df[col])):
            batch_tensor = df[col].values[row]
            print(batch_tensor)
            scores = process_tensor_str(batch_tensor)
            epoch_scores.append(scores)
        epoch_mean_scores.append(np.mean(epoch_scores))
        epoch_score_std.append(np.std(epoch_scores))
    
    print(epoch_mean_scores)
    print(epoch_score_std)

if __name__ == '__main__':
    process_tensor_str("tensor([0.2330, 0.2257, 0.2968, 0.1186, 0.1648, 0.2842, 0.1096])")
    main()