
import json
import sys 
import numpy as np
import pandas as pd 
#from InduceC45 import prepare_D

def main():
  argumentList = sys.argv[1:]
  csv_file = argumentList[0]
  json_file = argumentList[1]
  print("Training file {}".format(csv_file))
  print("Json file {}".format(json_file))

  json_dict = json_file_to_dict(json_file)
  D, A = prepare_D(csv_file, [])
  classifier(D,json_dict)

  total, data = classifier(D, json_dict)
  lst = data.values.tolist()

  total = data.values.sum()
  
  T = 0
  for j in range(len(lst)):
    T = T + lst[j][j]

  print("Total number of records classified: ", total)
  print("Total number of records correctly classified: ", T)
  print("Total number of records incorrectly classified: ", total - T)
  print("Overall Accuracy: ", T/total)
  print("Error Rate: ", (total-T)/total)


  # WHAT SHOULD n be for classofier ? 
  #cross_validation(D, A, 0, json_dict)
  ########## OLD #####################


def prepare_D(training_file, restrictions_list): 
  data = pd.read_csv('./' + training_file)
  D = pd.DataFrame(data) #See if you want to do a multi-hot encoding structure for ease of use 
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
  # print("D\n", D)
  A = list(D)
  # print("A\n", A)
  return D, A


def find_result(list_, dict_): 
  # print("dict", dict_)
  result = ""
  if "node" in  dict_.keys():
    node = dict_["node"]
    result = find_result(list_,node)
  elif "leaf" in  dict_.keys():
    return dict_['leaf']['decision']
    
  elif "edges" in  dict_.keys():
    edges = dict_["edges"]
    for edge in edges: 
      if edge['edge']['value'] in list_: 
        edge_of_interest = edge['edge']
        if "leaf" in edge_of_interest.keys():
          return  edge['edge']['leaf']['decision']
        else: 
          result = find_result(list_,edge_of_interest["node"])
          # print("result ", result)

  return result 




def matrix(a,b, result):
  zeros = []
  for i in range(len(result)):
    zeros.append([0]*len(result))


  actual = [] 
  classified = []
  for i in result:
    actual.append("Actual "+i)
    classified.append("Classified "+i)

  data = pd.DataFrame(data = zeros, index = actual, columns = classified)

  for i, j in zip(a, b):
    data["Classified "+i]["Actual "+j] = data["Classified "+i]["Actual "+j] + 1
  
  print('Matrix: ')
  print(data)
  
        
  return data

def classifier(D, raw):
  print("raw", raw)
  print("D", D)
  objects = list(D[D.columns[-1]])
  object_type = list(set(objects))

  overall = []
  records = len(D)
  for index, row in D.iterrows():
    passed_row = row.tolist()
    result = find_result(passed_row, raw)
    overall.append(result)

  #print("overall ", overall)

  x = D[D.columns[-1]].to_list()
  data = {'Predicted': overall, "Real": x}
  df = pd.DataFrame(data)

  # print("Result:{}\n".format(df))

  a = df['Predicted'].to_list()
  b = df['Real'].to_list()

  result = df['Real'].value_counts().index.to_list()
  print("result", result)
  result2 = df['Predicted'].value_counts().index.to_list()
  print("result2", result2)

  for i in result2:
    if (i not in result):
      result.append(i)

  data = matrix(a,b, result)
  return records, data

def json_file_to_dict(json_file): 
  with open(json_file) as f:
    data = json.load(f)
    return data 

main()