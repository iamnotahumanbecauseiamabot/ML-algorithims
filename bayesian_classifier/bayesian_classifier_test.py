from bayesian_classifier_train import *

features_probability = {'age':{}, 'menopause':{}, 'tumor_size':{}, 'inv_nodes':{},'node_caps':{}, 'deg_malig':{}, 'breast':{}, 'breast_quad':{}, 'irradiat':{}}
features_probability_no = make_dic(features_probability,no_df, num_no)
features_probability = {'age':{}, 'menopause':{}, 'tumor_size':{}, 'inv_nodes':{},'node_caps':{}, 'deg_malig':{}, 'breast':{}, 'breast_quad':{}, 'irradiat':{}}
features_probability_yes = make_dic(features_probability,yes_df, num_yes)

df = pd.read_csv('breast-cancer_test.txt', sep = ',')

acc_list = []
for _, row in df.iterrows():
    pro_no = calc_bayes(features_probability_no, row) * (num_no/(num_no + num_yes))
    pro_yes = calc_bayes(features_probability_yes, row)* (num_yes/(num_no + num_yes))
    #print(pro_no, pro_yes)
    if pro_no > pro_yes:
        if row['Class'] == 'no-recurrence-events':
            acc_list.append(1)
        else:
            acc_list.append(0)
    elif pro_no < pro_yes:
        if row['Class'] == 'recurrence-events':
            acc_list.append(1)
        else:
            acc_list.append(0)
    else:
        acc_list.append(1)


#print accuracy
print('{:.5}'.format( (sum(acc_list)/len(acc_list))*100))

