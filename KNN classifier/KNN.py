import scipy.io
import math
import numpy as np
from operator import itemgetter
from collections import Counter

mnist = scipy.io.loadmat('./USPS.mat')

test_data = mnist['test_data'].T
test_lbl = mnist['test_lbl']
train_data = mnist['train_data'].T
train_lbl = mnist['train_lbl']

def mode(labels):
    return Counter(labels).most_common(1)[0][0]
  
def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)
  

def KNN(test_point, data_distribution = train_data, data_distribution_labels = train_lbl, k=5):
    
    distance_label = []
    
    for data, label in zip(data_distribution, data_distribution_labels):
        
        distance = euclidean_distance(data, test_point)
        distance_label.append( [distance, label[0]] )
        
    sort_list = sorted(distance_label, key=itemgetter(0))
        
    get_top_k_labels = sort_list[:k]

    get_top_k_labels = np.asarray(get_top_k_labels)[:, 1]
    
    class_predicted = mode(get_top_k_labels)
    
    return class_predicted
  

def test_acc(test_data_points=test_data, test_data_points_labels=test_lbl):
  
  acc_list = []
  ind = 0
  for data_point, data_label in zip(test_data_points, test_data_points_labels):
    class_pred = KNN(data_point)
    if data_label[0]==class_pred:
      acc_list.append(1)
    else:
      acc_list.append(0)
    
    if ind%1==0:
      print(ind)
    
    ind+=1
  
  return sum(acc_list)/len(acc_list)

acc = test_acc()
    
