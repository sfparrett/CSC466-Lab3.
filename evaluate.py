import sys 
import pandas as pd 
import math 
import json 
import numpy as np
from utils import *

def main():
  # python3 validation.py <TrainingSetFile.csv> [<restrictionsFile>]

  # print("Select a number indicating the training file:")
  # print("\t 1 = openHouses.csv\n\t 2 = adult-stretch.csv\n\t 3 = adult+stretch.csv\n\t 4 = yellow-small.csv\n\t 5 = yellow-small+adult-stretch.csv \n\t 6 = agaricus-lepiota.csv \n\t 7 = nursery.csv \n")
  # training_file = int(input(""))
  # if training_file == 1: 
  #   training_file = "openHouses.csv" 
  # elif training_file == 2: 
  #   training_file = "adult-stretch.csv"
  # elif training_file == 3: 
  #   training_file = "adult+stretch.csv"
  # elif training_file == 4: 
  #   training_file = "yellow-small.csv"
  # elif training_file == 5: 
  #   training_file = "yellow-small+adult-stretch.csv"
  # elif training_file == 6: 
  #   training_file = "agaricus-lepiota.csv"
  # else: 
  #     training_file = "nursery.csv" 

  # print("training file {}".format(training_file))

  # #restriction_file = input("Restrictions File: ")
  # threshold = float(input("Threshold: "))
  # n = int(input("n folds: "))

    # clean None v
  #training_file = "Extra_Example.csv"
  
    # argumentList = sys.argv[1:]
    # training_file = argumentList[0]
    

    # if len(argumentList) > 1: 
    #     restrictions_file = argumentList[1]
    #     my_file = open(restrictions_file, "r")
    #     content = my_file.read()
    #     restrictions_list = content.split(",")
    #     my_file.close()

    # n = float(input("Input n value: "))
    # threshold = float(input("Input threshold value: "))
    # # print(type(threshold))

  #t = input("Which type: 1 = C45, 2 = KNN, 3 = Forest:")

  #training_file = "winequality-red-fixed.csv"
  training_file = "iris.data.csv"

  #For C45
  threshold = 0.1
  n = 3 #amount of folds 

  #For Random Trees
  m = .8
  k_rt = .9
  N = 3

  #KNN 
  k_knn = 3
  
  restrictions_list = []

  is_numeric, D, A = prepare_D(training_file, restrictions_list)

  # CLEAN DATA SET 
  D.dropna()
  for i in D.columns:
    D.drop(D.loc[D[i] == "?"].index, inplace=True)

  #cross_validation(is_numeric, D, A, n, threshold, m, k_rt, N, k_knn, t)
  cross_validation2(is_numeric, D, A, n, threshold)
  # forest = randomForest(D=D, A=A, N=3, m=int(round(.80*len(A))), k=int(round(.9*len(D))), DecisionTreeImplementation="C45", is_numeric=is_numeric)
  # print("forest", forest[0].d)
  # c = RFClassify(forest, x= [6.7,3.0,5.0,1.7,"Iris-versicolor"])
  # print("class", c)



main()

# def prepare_D(training_file, restrictions_list): 
#     data = pd.read_csv('./' + training_file)
#     D = pd.DataFrame(data) #See if you want to do a multi-hot encoding structure for ease of use 
#     value = D.iloc[1][0]
#     numbers_list = D.copy()
#     numbers_list = list(numbers_list.iloc[0])
#     D = D.iloc[2:]
#     A = list(D)
#     ignore_list = []
#     ignore_list.append(value)

#     for i in range(len(numbers_list)):
#         if type(numbers_list[i]) != float: 
#             if int(numbers_list[i]) == -1: 
#                 ignore_list.append(A[i])

#         if len(restrictions_list) > 0: 
#             if (int(restrictions_list[i]) == 0): 
#                 ignore_list.append(A[i])
#         i+=1 

#     D = D[[c for c in D if c not in ignore_list] + [value]]
#     # print("D\n", D)
#     A = list(D)
#     # print("A\n", A)
#     return D, A

# def find_result(list_, dict_): 
#   result = ""
#   if "node" in  dict_.keys():
#     node = dict_["node"]
#     result = find_result(list_,node)
#   elif "leaf" in  dict_.keys():
#     return dict_['leaf']['decision']
    
#   elif "edges" in  dict_.keys():
#     edges = dict_["edges"]
#     for edge in edges: 
#       if edge['edge']['value'] in list_: 
#         edge_of_interest = edge['edge']
#         if "leaf" in edge_of_interest.keys():
#           return  edge['edge']['leaf']['decision']
#         else: 
#           result = find_result(list_,edge_of_interest["node"])

#   return result 

# def zero_matrix(result):
#   zeros = []
#   for i in range(len(result)):
#     zeros.append([0]*len(result))

#   actual = [] 
#   classified = []
#   for i in result:
#     actual.append("Actual "+i)
#     classified.append("Classified "+i)
  
#   data = pd.DataFrame(data = zeros, index = actual, columns = classified)

#   return data


# def matrix(a,b, result):
#   data = zero_matrix(result)

#   for i, j in zip(a, b):
#     data["Classified "+i]["Actual "+j] = data["Classified "+i]["Actual "+j] + 1

#   return data

# def classifier(D, raw):
#   objects = list(D[D.columns[-1]])
#   object_type = list(set(objects))

#   overall = []
#   records = len(D)
#   for index, row in D.iterrows():
#     passed_row = row.tolist()
#     result = find_result(passed_row, raw)
#     overall.append(result)

#   x = D[D.columns[-1]].to_list()
#   data = {'Predicted': overall, "Real": x}
#   df = pd.DataFrame(data)

#   # print("Result:{}\n".format(df))

#   a = df['Predicted'].to_list()
#   b = df['Real'].to_list()

#   result = df['Real'].value_counts().index.to_list()
#   result2 = df['Predicted'].value_counts().index.to_list()

#   for i in result2:
#     if (i not in result):
#       result.append(i)

#   data = matrix(a,b, result)

  
#   return records, data, result

# def cross_validation(D, A, n, threshold):
#   matrix = []
#   D = D.sample(frac = 1).reset_index(drop=True)
#   overall = len(D)
#   o_eval = len(D)
#   correct = 0 
#   total_amt = 0 
#   accuracy = []

#   if(n == 0):
#     n = 1
#   elif(n == -1):
#     n = len(D)

#   i = 1 
#   while i <= n:
#     num = round(overall/n)
#     if (num > o_eval or (i - n) == 0):
#       num = o_eval
#     o_eval = o_eval - num
#     end = total_amt + num - 1
#     if (total_amt+num) > overall:
#       end = overall

#     send_rec = D.loc[total_amt:end].reset_index(drop=True)

#     if (n == 1):
#       send_C45 = send_rec
#     else:
#       send_C45 = D.drop(range(total_amt,end)).reset_index(drop=True)

#     before_node = Normal()
#     dict_ = {}
#     Attributes = A.copy()

#     node, final_dict = C45(send_C45, Attributes, threshold, before_node, dict_)

#     total, data, result = classifier(send_rec, final_dict)
#     lst = data.values.tolist()

#     matrix.append(data)

#     total = data.values.sum()
    
#     T = 0
#     for j in range(len(lst)):
#       T = T + lst[j][j]

#     total_amt += total
#     correct += T
#     accuracy.append(T/total)
#     i +=1
  

# #   print()
#   print("Overall Matrix: ")

#   overall_matrix = zero_matrix(result)
#   for i in matrix:
#     overall_matrix = overall_matrix.add(i, fill_value=0)
#   print(overall_matrix)

#   print()
#   print("Totals: ")
#   #print("Total Number of records classified: ", total_amt)
#   print("Overall Accuracy: ", correct/total_amt)
#   print("Individual accuracies: ", accuracy)
#   print("Average accuracy: ", sum(accuracy)/len(accuracy))


# def json_file_to_dict(json_file): 
#   with open(json_file) as f:
#     data = json.load(f)
#     return data 


# def C45(D, A, threshold, node, dict_):
# #   print("threshold 2", threshold)
#   best = []
#   bol, attr = check_home(D, A)

#   if (bol == True): 
#     leaf = Leaf(1,attr)
#     if type(attr) == tuple: 
#       a = attr[0]
#     else: 
#       a = attr 
#     leaf_dict = {"leaf": {"decision": a, "p":1}}
#     return leaf , leaf_dict   

#   elif not A:
#     # print("Not Homogen 1")
#     c, p = find_most_frequent_label(D)
#     leaf = Leaf(1,c)
#     leaf_dict = {"leaf": {"decision": c, "p": p}}
#     return leaf , leaf_dict 
    
#   else:
#     # print("Not Homogen 2")
#     Ag = selectSplittingAttribute(A,D,threshold)
#     if Ag == None:
#       c, p = find_most_frequent_label(D)
#       leaf = Leaf(1,c)
#       leaf_dict = {"leaf": {"decision": c, "p": p}}
#       return leaf, leaf_dict 

#     else:
#       node = Normal(Ag)
#       node_dict = {"node": { "var": Ag,"edges": []}}

#       labels = list(D[Ag])
#       option_labels = list(set(labels))
#       A.remove(Ag)

#       for i in range(len(option_labels)):
#         try:
#           data = D.loc[(D[Ag] == option_labels[i])] 
#         except:
#           continue
#         else:
#           child, child_dict = C45(data, A, threshold, node, dict_) 
#           edge = Edge(option_labels[i], child)

#           edge_dict = {"edge": {"value": option_labels[i], "node": child_dict}}
#           node.edges.append(edge)
#           node_dict["node"]['edges'].append(edge_dict)
#         #   print("threshold 3", threshold)

#   # print("tree {}".format(node_dict))
#   return node, node_dict



# class Leaf:
#     def __init__(self,probability, result):
#       self.probability = probability
#       self.result = result
  
#     def __repr__(self):
#       ret = "{} {}".format( self.probability, self.result)
#       return ret 

# class Normal:
#     def __init__(self, A=None):
#       self.A = A
#       self.edges = []
  
#     def __repr__(self):
#       ret1 = "{}".format( self.A)
#       ret2 = ""
#       for edge in self.edges: 
#         ret2 += "{}".format( edge.__repr__())
#       return ret1 + ret2  

# class Edge:
#     def __init__(self, label, node):
#       self.label = label 
#       self.node = node  # Normal or Edge 
  
#     def __repr__(self):
#       ret = "{} {}".format( self.label,  self.node.__repr__())
#       return ret 
      

# def find_most_frequent_label(D): 
#   # print("most frequent label")
#   # print("D\n", D)
#   df = D.apply(pd.Series.value_counts)
#   # print("df\n", df)
#   max_column = df.max().sort_values(ascending=False).index[0]
#   # print("max column\n", max_column)
#   label = df[max_column].idxmax()
#   percentage = df[max_column].max()/df[max_column].sum()
#   # print("label", df[max_column].max())
#   # print("label 2", df[max_column].sum())
#   return label, percentage



# def selectSplittingAttribute(A,D,threshold):
#   gain = []
#   p0, option_labels = first_entropy(D, D.columns[-1])
#   for i in range(len(A)):
#     pA = entropy(D, A[i], D.columns[-1])
#     gain.append(p0 - pA) 
#   best = max(gain)
# #   print("best", best)
# #   print(type(best))

# #   print("threshold", threshold)
# #   print(type(threshold))
#   # print("Best: ", best)
#   if best > threshold:
#     x = A[gain.index(max(gain))]
#     return x
#   else:
#     return None


# def first_entropy(D, A):
#   labels = list(D[A])
#   option_labels = list(set(labels))
#   total = len(D)
#   entropy = 0 
#   for i in range (len(option_labels)):  
#     first_ratio = int((D[A] == option_labels[i]).sum()) / total 
#     entropy = entropy + log(first_ratio)
#   return entropy, option_labels 

# def entropy(D, a, A):
#   entropy1, option_labels = first_entropy(D, A)
#   if (a == A):
#     return entropy1
#   labels2 = list(D[a])
#   option_labels2 = list(set(labels2))
#   overall = 0
#   for i in range (len(option_labels2)): 
#     entropy = 0
#     for j in range (len(option_labels)): 
#       label = option_labels2[i]
#       total = int((D[a] == label).sum())
#       ratio_1 = len(D.loc[(D[a] == label) & (D[A] == option_labels[j]) ].index) / total
#       entropy = entropy + log(ratio_1)
#     overall = overall + (total/len(D[a])) * entropy
#   return overall


# def log(A):
#   try:
#     x = math.log2(A)
#   except:
#     x = 0
#   return -((A) * x)


# def check_home(D, A): 
#   d = D[[D.columns[-1]]]
#   results = d.value_counts().index.to_list()
#   if len(results) > 1: 
#     return False, False 
#   else: 
#     return True, results[0]

# main()

