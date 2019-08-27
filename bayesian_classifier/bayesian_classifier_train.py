import pandas as pd
import numpy as np
import scipy.io

df = pd.read_csv('breast-cancer_train.txt', sep = ',')


no_df = df[df.Class == 'no-recurrence-events']
yes_df = df[df.Class == 'recurrence-events']
num_no = len(no_df)
num_yes = len(yes_df)


def make_dic(dic_input, df_input, num_input):
    for j in dic_input:
        s = df_input.groupby(j).count().iloc[:,0]
        dic = {}
        num_absent = 0
        for i in s.keys():
            if str(i) in '?':
                num_absent = s[i]
        for i in s.keys():
            if str(i) not in '?':
                dic[str(i)] = s[i]/(num_input - num_absent)
        dic_input[j] = dic
    return dic_input

def calc_bayes(features_probability_input, row_to_iter):
    train_features = ['age', 'menopause', 'tumor_size', 'inv_nodes','node_caps', 'deg_malig', 'breast', 'breast_quad', 'irradiat']
    bayes_probability = 1
    for f in train_features:
        try:
            bayes_probability *= features_probability_input[f][str(row_to_iter[f])]
        except KeyError:
            return 0
    return bayes_probability

