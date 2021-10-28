import sys 
import pandas as pd 
import math 
import json 
import numpy as np
from utils import *
import itertools
from timeit import default_timer as timer

def get_key(val, my_dict):
    lst = []
    for key, value in my_dict.items():
         if val == value:
             lst.append(key)
    if not lst:
      return None
    return lst

def main():
  print("Select a number indicating the training file:")
  print("\t 1 = iris.data.csv\n\t 2 = letter-recognition.data\n\t 3 = winequality-red-fixed.csv\n\t 4 = winequality-white-fixed.csv\n\t 5 = crx.data.csv \n\t 6 = heart.csv\n")
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
  else: 
      training_file = "heart.csv" 

  print("Threshold: (0-1)")
  thresholds = [float(input(""))]
  print("Folds (1-10) & -1 for all but one: ")
  folds = [float(input(""))]     # [x for x in range(0, 20, 3)]
  print("\n\n\nC45 {}: ".format(training_file))
  
  restrictions_list = []
  is_numeric, D, A = prepare_D(training_file, restrictions_list)
  
  D.dropna()
  for i in D.columns:
    D.drop(D.loc[D[i] == "?"].index, inplace=True)

  keys = get_key(True, is_numeric)
  if get_key(True, is_numeric) != None:
    for i in keys:
      D[i] = pd.to_numeric(D[i])

  for x in itertools.product(thresholds, folds):
    if x[0] == -1: 
      fold= len(D)
    else: 
      fold = x[0]

    print("\n\nThreshold = {}".format(fold))
    print("Number of folds = {}".format(x[1]))
    n = x[1]
    threshold = x[0]
   
    start = timer()
    cross_validation2(is_numeric, D, A, n, threshold)
    end = timer()
    print("Time= {}s".format(end-start))

main()




