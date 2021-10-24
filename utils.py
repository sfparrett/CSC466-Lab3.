import sys 
import pandas as pd 
import math 
import json 
import numpy as np

def prepare_D(training_file, restrictions_list): 
    data = pd.read_csv(training_file)
    D = pd.DataFrame(data) #See if you want to do a multi-hot encoding structure for ease of use 
    n = is_numeric(D)
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
    # print("class in D before C45\n ", D["class"].unique())

    return n, D, A



def is_numeric(D): 
  numeric = {}
  for col in D.columns:
    if int(D.at[0,col]) == 0: 

      numeric[col] = True 
    else: 
      numeric[col] = False 

  return numeric


def find_result(list_, dict_): 
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

  return result 

def zero_matrix(result):
  zeros = []
  for i in range(len(result)):
    zeros.append([0]*len(result))

  actual = [] 
  classified = []
  for i in result:
    actual.append("Actual "+i)
    classified.append("Classified "+i)
  
  data = pd.DataFrame(data = zeros, index = actual, columns = classified)

  return data


def matrix(a,b, result):
  data = zero_matrix(result)

  for i, j in zip(a, b):
    data["Classified "+i]["Actual "+j] = data["Classified "+i]["Actual "+j] + 1

  return data

def classifier(D, raw):
#   print("classifier func", D["class"].unique()) 
  objects = list(D[D.columns[-1]])
  object_type = list(set(objects))

  overall = []
  records = len(D)
  for index, row in D.iterrows():
    passed_row = row.tolist()
    result = find_result(passed_row, raw)
    overall.append(result)

  x = D[D.columns[-1]].to_list()
  data = {'Predicted': overall, "Real": x}
  df = pd.DataFrame(data)
#   print("Classifier func D\n", df)
#   print("Real", df["Real"].unique()) 
#   print("Predicted", df["Predicted"].unique())
#   exit()

  # print("Result:{}\n".format(df))

  a = df['Predicted'].to_list()
  b = df['Real'].to_list()

  result = df['Real'].value_counts().index.to_list()
  result2 = df['Predicted'].value_counts().index.to_list()

  for i in result2:
    if (i not in result):
      result.append(i)

  data = matrix(a,b, result)

  
  return records, data, result

def cross_validation(is_numeric, D, A, n, threshold):
#   print("cross val func", D["class"].unique()) 
  matrix = []
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
      send_C45 = send_rec
    else:
      send_C45 = D.drop(range(total_amt,end)).reset_index(drop=True)

    before_node = Normal()
    dict_ = {}
    Attributes = A.copy()

    node, final_dict = C45(is_numeric, send_C45, Attributes, threshold, before_node, dict_)

    total, data, result = classifier(send_rec, final_dict)
    lst = data.values.tolist()

    matrix.append(data)

    total = data.values.sum()
    
    T = 0
    for j in range(len(lst)):
      T = T + lst[j][j]

    total_amt += total
    correct += T
    accuracy.append(T/total)
    i +=1
  

#   print()
  print("Overall Matrix: ")

  overall_matrix = zero_matrix(result)
  for i in matrix:
    overall_matrix = overall_matrix.add(i, fill_value=0)
  print(overall_matrix)

  print()
  print("Totals: ")
  #print("Total Number of records classified: ", total_amt)
  print("Overall Accuracy: ", correct/total_amt)
  print("Individual accuracies: ", accuracy)
  print("Average accuracy: ", sum(accuracy)/len(accuracy))


def json_file_to_dict(json_file): 
  with open(json_file) as f:
    data = json.load(f)
    return data 


def C45(is_numeric, D, A, threshold, node, dict_):
#   print("threshold 2", threshold)
  best = []
  bol, attr = check_home(D, A)

  if (bol == True): 
    leaf = Leaf(1,attr)
    if type(attr) == tuple: 
      a = attr[0]
    else: 
      a = attr 


    leaf_dict = {"leaf": {"decision": a, "p":1}}
    return leaf , leaf_dict   

  elif not A:
    # print("Not Homogen 1")
    c, p = find_most_frequent_label(D)
    leaf = Leaf(1,c)

    leaf_dict = {"leaf": {"decision": c, "p": p}}
    return leaf , leaf_dict 
    
  else:
    # print("Not Homogen 2")
    Ag = selectSplittingAttribute(is_numeric, A,D,threshold)
    if Ag == None:
      c, p = find_most_frequent_label(D)
      leaf = Leaf(1,c)

      leaf_dict = {"leaf": {"decision": c, "p": p}}
      return leaf, leaf_dict 

    else:
      node = Normal(Ag)
      node_dict = {"node": { "var": Ag,"edges": []}}

      labels = list(D[Ag])
      option_labels = list(set(labels))
      A.remove(Ag)

      for i in range(len(option_labels)):
        try:
          data = D.loc[(D[Ag] == option_labels[i])] 
        except:
          continue
        else:
          child, child_dict = C45(is_numeric, data, A, threshold, node, dict_) 
          edge = Edge(option_labels[i], child)

          # EDGE CHANGES BASED ON NUMERIC 
          # if is_numeric 

          edge_dict = {"edge": {"value": option_labels[i], "node": child_dict}}
          node.edges.append(edge)
          node_dict["node"]['edges'].append(edge_dict)
        #   print("threshold 3", threshold)

  # print("tree {}".format(node_dict))
  return node, node_dict

class Leaf:
    def __init__(self,probability, result):
      self.probability = probability
      self.result = result
  
    def __repr__(self):
      ret = "{} {}".format( self.probability, self.result)
      return ret 

class Normal:
    def __init__(self, A=None):
      self.A = A
      self.edges = []
  
    def __repr__(self):
      ret1 = "{}".format( self.A)
      ret2 = ""
      for edge in self.edges: 
        ret2 += "{}".format( edge.__repr__())
      return ret1 + ret2  

class Edge:
    def __init__(self, label, node):
      self.label = label 
      self.node = node  # Normal or Edge 
  
    def __repr__(self):
      ret = "{} {}".format( self.label,  self.node.__repr__())
      return ret 

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

def find_most_frequent_label(D): 
  # print("most frequent label")
  # print("D\n", D)
  df = D.apply(pd.Series.value_counts)
  # print("df\n", df)
  max_column = df.columns[-1]
  # print("max column\n", max_column)
  label = df[max_column].idxmax()
  percentage = df[max_column].max()/df[max_column].sum()
  # print("label", df[max_column].max())
  # print("label 2", df[max_column].sum())
  return label, percentage

def selectSplittingAttribute(is_numeric, A,D,threshold):
  
  gain = []
  for a in A: 
    gain.append(findBestSplit(a, D))

  best = max(gain)

  # NOT SURE IF THIS CODE STILL STANDS 
  if best > threshold:
    x = A[gain.index(max(gain))]
    return x
  else:
    return None

  
def info_gain_numeric(a, D):
  # HOW DO YOU FIND THIS 
  pass  

def findBestSplit(is_numeric, a, D):

  if is_numeric[a]: 
    unique = D[a].unique()
    all_info_gains = []
    for num in unique: 
      gain = info_gain_numeric(a, D)
      all_info_gains.append(gain)

    return max(all_info_gains)

  else: 
    p0, option_labels = first_entropy(D, D.columns[-1])
    pA = entropy(D, a, D.columns[-1])
    return p0 - pA


  exit()






def selectSplittingAttribute_part1(A,D,threshold):
  gain = []
  p0, option_labels = first_entropy(D, D.columns[-1])
  for i in range(len(A)):
    pA = entropy(D, A[i], D.columns[-1])
    gain.append(p0 - pA) 
  best = max(gain)

  if best > threshold:
    x = A[gain.index(max(gain))]
    return x
  else:
    return None


# def findBestSplit(a, D):
#   #  Questions: 
#   # 1.) What does ascending order look like for D
#   # 2.) what is alpha 
#   counts = [[]]
#   Gain = []
#   alpha = []
#   p0 = first_entropy(D, A)
#   D = D.sort_values(by=[a], ascending = True) # ascending = True or False?
#   print(D)
#   for i in range(1, len(D)): 
#     d = D[a][i] # go into data point , check this 
#     print("d", d)
    
#     for j in range(k): 
#       j+=1 
#     i+=1 


#   pass 

def first_entropy(D, A):
  labels = list(D[A])
  option_labels = list(set(labels))
  total = len(D)
  entropy = 0 
  for i in range (len(option_labels)):  
    first_ratio = int((D[A] == option_labels[i]).sum()) / total 
    entropy = entropy + log(first_ratio)
  return entropy, option_labels 

def entropy(D, a, A):
  entropy1, option_labels = first_entropy(D, A)
  if (a == A):
    return entropy1
  labels2 = list(D[a])
  option_labels2 = list(set(labels2))
  overall = 0
  for i in range (len(option_labels2)): 
    entropy = 0
    for j in range (len(option_labels)): 
      label = option_labels2[i]
      total = int((D[a] == label).sum())
      ratio_1 = len(D.loc[(D[a] == label) & (D[A] == option_labels[j]) ].index) / total
      entropy = entropy + log(ratio_1)
    overall = overall + (total/len(D[a])) * entropy
  return overall

def log(A):
  try:
    x = math.log2(A)
  except:
    x = 0
  return -((A) * x)

def check_home(D, A): 
  d = D[[D.columns[-1]]]
  results = d.value_counts().index.to_list()
  if len(results) > 1: 
    return False, False 
  else: 
    return True, results[0]
