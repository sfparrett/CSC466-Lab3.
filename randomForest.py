import sys 
import pandas as pd 
import math 
import json 
import numpy as np
import random
from collections import Counter
from utils import *
import itertools
from timeit import default_timer as timer

def randomForest(D, A, N, m, k, is_numeric, class_labels):
  forest = []
  m = int(round(m*len(A)))
  k = int(round(k*len(D)))

  for i in range(0,N): 
    AttList = random.sample(A, m)
    Data =  D.sample(n = k)
    f = Forest(AttList, Data, 0, is_numeric, class_labels)
    forest.append(f)
    i+=1
    
  return forest

def most_frequent(list_):
  return max(set(list_), key = list_.count)

def RFClassify(Forest, x, D):
  prediction = []
  for f in Forest: 
    prediction.append( find_result(f.A, x, f.d, D))
  
  return most_frequent(prediction)
  
class Forest: 
  def __init__(self, A, D,threshold, is_numeric, class_labels): 
    self.A = A
    self.D = D
    self.threshold = threshold
    node = Normal()
    d = {}
    T, d = C45(is_numeric, D, A,threshold,node,d, class_labels)
    self.d = d


def get_key(val, my_dict):
    lst = []
    for key, value in my_dict.items():
         if val == value:
             lst.append(key)
    if not lst:
      return None
    return lst

def classifier_tree(D, Forest, class_labels):
  objects = list(D[D.columns[-1]])
  attributes = list(D.iloc[:1])

  overall = []
  overall_r = []
  o_data = []
  records = len(D)

  for index, row in D.iterrows():
    passed_row = row.tolist()
    if len(attributes) != len(passed_row): 
      print("ERROR")
      sys.exit()
    result = RFClassify(Forest, passed_row, D)
    overall.append(result)


  x = D[D.columns[-1]].to_list()
  data = {'Predicted': overall, "Real": x}
  df = pd.DataFrame(data)

  df.dropna()

  a = df['Predicted'].to_list()
  b = df['Real'].to_list()

  result = df['Real'].value_counts().index.to_list()

  data = matrix(a,b, result, class_labels)
  o_data.append(data)
  overall_r.append(result)

  return records, data, result 

def cross_validation_tree(D, A, n, m, k, is_numeric):
  D = D.sample(frac = 1).reset_index(drop=True)
  overall = len(D)
  o_eval = len(D)
  correct = 0
  total_amt = 0
  accuracy = []

  if(n == 0):
    n = 1
  elif(n == -1):
    n = len(D)

  i = 1 
  result = []
  matrix_fin = []

  while i <= n:
    num = round(overall/n)
    if (num > o_eval or (i - n) == 0):
      num = o_eval
    o_eval = o_eval - num
    end = total_amt + num - 1
    if (total_amt+num) > overall:
      end = overall

    send_rec = D.loc[total_amt:end].reset_index(drop=True)
    

    if (n == 1):
      send_forest = send_rec
    else:
      send_forest= D.drop(range(total_amt,end)).reset_index(drop=True)

    class_label = D[D.columns[-1]].value_counts().to_dict()
    
    forest = randomForest(send_forest, A, n, m, k, is_numeric,  class_label) 
    i+=1

    total, data, result = classifier_tree(send_rec, forest, class_label)
    matrix_fin = []
    lst = data.values.tolist()
    total = data.values.sum()

    T = 0
    for j in range(len(lst)):
      T = T + lst[j][j]

    correct += T
    accuracy.append(T/total)
    matrix_fin.append(data)
    total_amt += total

  overall_matrix = zero_matrix(result)
  for j in matrix_fin:
    overall_matrix = overall_matrix.add(j, fill_value=0)
  print(overall_matrix)

  print("Totals: ")
  #print("Total Number of records classified: ", total_amt)
  print("Overall Accuracy: ", correct/total_amt)
  print("Individual accuracies: ", accuracy)
  print("Average accuracy: ", sum(accuracy)/len(accuracy))

def main():
    print("Select a number indicating the training file:")
    print("\t 1 = iris.data.csv\n\t 2 = letter-recognition.data\n\t 3 = winequality-red-fixed.csv\n\t 4 = winequality-white-fixed.csv\n\t 5 = crx.data.csv \n\t 6 = nursery.csv\n\t 7 = agaricus-lepiota.csv\n\t 8 = heart.csv\n")
    training_file = int(input(""))
    if training_file == 1: 
        training_file = "iris.data.csv" 
    elif training_file == 2: 
        training_file = "letter-recognition.data.csv"
    elif training_file == 3: 
        training_file = "winequality-red-fixed.csv"
    elif training_file == 4: 
        training_file = "winequality-white-fixed.csv"
    elif training_file == 5: 
        training_file = "crx.data.csv"
    elif training_file == 6: 
        training_file = "nursery.csv"
    elif training_file == 7: 
        training_file = "agaricus-lepiota.csv"
    else: 
        training_file = "heart.csv" 

    print("Percentage of Attribute (0 - 1)")
    m = [float(input(""))] #attributes
    print("Percentage of Data (0 - 1)")
    k = [float(input(""))] #data
    folds = [10] #needs to be set to this  

    print("\n\n\nRandom Forest {}: ".format(training_file))
    restrictions_list = []
    is_numeric, D, A = prepare_D(training_file, restrictions_list)
    
    D.dropna()
    for i in D.columns:
        D.drop(D.loc[D[i] == "?"].index, inplace=True)

    keys = get_key(True, is_numeric)
    if get_key(True, is_numeric) != None:
        for i in keys:
            D[i] = pd.to_numeric(D[i])

   

    for x in itertools.product(m, k):
        m = x[0]
        k = x[1]
        print("\n\nPercentage of Attributes = {}".format(m))
        print("Percentage of Data= {}".format(k))
        print("Number of folds = {}".format(folds[0]))
     
        start = timer()
        cross_validation_tree(D, A, folds[0], m, k, is_numeric)
        end = timer()
        print("Time= {}s".format(end-start))

main()
