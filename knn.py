import sys 
import pandas as pd 
import math 
import json 
import numpy as np
import random

from pandas.core.tools.numeric import to_numeric
from utils import *


def is_num(D): 
  num = []
  cat = []
  for col in D.columns:
    if int(D.at[0,col]) == 0: 
        num.append(col)
    else: 
        cat.append(col)
  return (num, cat)

def prepare_D_knn(training_file, restrictions_list): 
    data = pd.read_csv(training_file)
    D = pd.DataFrame(data) #See if you want to do a multi-hot encoding structure for ease of use 
    n = is_num(D)
    value = D.iloc[1][0]
    numbers_list = D.copy()
    numbers_list = list(numbers_list.iloc[0])
    D = D.iloc[2:]
    A = list(D)
    ignore_list = []
    ignore_list.append(value)

    for i in range(len(numbers_list)):
        if type(numbers_list[i]) != float: 
            if int(numbers_list[i]) == -1: 
                ignore_list.append(A[i])

        if len(restrictions_list) > 0: 
            if (int(restrictions_list[i]) == 0): 
                ignore_list.append(A[i])
        i+=1 

    D = D[[c for c in D if c not in ignore_list] + [value]]
    A = list(D)
    # print("A\n", A)
    # print("class in D before C45\n ", D["class"].unique())

    return n, D, A

def classifier_knn(D, k, is_numeric):
    D = D.reset_index(drop=True)

    D_num = D[is_numeric[0]]
    D_cat =  D[is_numeric[1][:-1]]

    D_num = D_num.apply(to_numeric)

    D_num = (D_num - D_num.min())/(D_num.max()-D_num.min())
    
    predicted = []
    for i in range(len(D)):
        if not D_cat.empty:
            a = np.array(D_cat[D_cat.index==i])[0]
            cat = np.sum(np.array(D_cat) != a, axis =1)/(len(is_numeric[1]) -1)
        else:
            cat = 0 
        if not D_num.empty:
            a = np.array(D_num[D_num.index==i])[0]
            #num = np.linalg.norm(np.array(D_num) - a)
            num = np.sqrt(np.sum(np.square(np.subtract(np.array(D_num), a)), axis =1))
        else:
            num = 0

        dist = (cat + num) 
        x = D[D.columns[-1]].to_list()

        data = {'Distance': dist, "Real": x}
        df = pd.DataFrame(data)
        df = df.sort_values(by = ['Distance']).reset_index(drop=True)
        # print(df[1:k+1]['Real'].value_counts())
        prediction = df[1:k+1]['Real'].value_counts().index[0]
        predicted.append(prediction)

    data = {'Predicted': predicted, "Real": x}
    df = pd.DataFrame(data)

    a = df['Predicted'].to_list()
    b = df['Real'].to_list()

    result = df['Real'].value_counts().index.to_list()

    data = matrix(a,b, result)

    records = len(D)
    
    return records, data, result 
 
def main():
    #training_file = "winequality-red-fixed.csv"
    #training_file = "iris.data.csv"
    training_file = "crx.data.csv"
    
    #training_file = "letter-recognition.data.csv"
    
    #KNN 
    k = 10
    
    restrictions_list = []

    is_numeric, D, A = prepare_D_knn(training_file, restrictions_list)

    # CLEAN DATA SET 
    D.dropna()
    for i in D.columns:
        D.drop(D.loc[D[i] == "?"].index, inplace=True)


    #cross validation

    correct = 0
    total_amt = 0
    accuracy = []

    for i in range(2,k):
        records, data, result = classifier_knn(D, i, is_numeric)
        matrix_fin = []
    
        lst = data.values.tolist()

        total = data.values.sum()

        T = 0
        for j in range(len(lst)):
            T = T + lst[j][j]

        correct = T
        accuracy.append(T/total)
        matrix_fin.append(data)
        total_amt += total

        print()
        print("Overall Matrix of: KNN - K =", i)

        overall_matrix = zero_matrix(result)
        for j in matrix_fin:
            overall_matrix = overall_matrix.add(j, fill_value=0)
        print(overall_matrix)

        print()
        print("Totals: ")
        #print("Total Number of records classified: ", total_amt)
        print("Overall Accuracy: ", correct/total_amt)
        print("Individual accuracies: ", accuracy)
        print("Average accuracy: ", sum(accuracy)/len(accuracy))


            

main()
