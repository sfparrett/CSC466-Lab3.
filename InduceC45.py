
import sys 
import pandas as pd 
import math 
import json 

def main():
    # python3 InduceC45 <TrainingSetFile.csv> [<restrictionsFile>]
  
  argumentList = sys.argv[1:]
  training_file = argumentList[0]
  restrictions_list = []

  if len(argumentList) > 1: 
    restrictions_file = argumentList[1]
    my_file = open(restrictions_file, "r")
    content = my_file.read()
    restrictions_list = content.split(",")
    my_file.close()

  main_logic(training_file,restrictions_list)


def main_logic(training_file, restrictions_list): 
    # print("Training file {}".format(training_file))
  # print("Restrictions list {}".format(restrictions_list))
  threshold = float(input("Input Threshold Value: "))
  json_filename = input("Please enter name for JSON file: ")

  D, A = prepare_D(training_file, restrictions_list)
  print("D", D)
  print("A",A)
  node = Normal()
  empty_dict = {}
  tree, dictionary = C45(D, A, threshold, node, empty_dict)
  f = open(json_filename, "w")
  json.dump(dictionary, f)
  f.close()



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


def C45(D, A, threshold, node, dict_):
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
    Ag = selectSplittingAttribute(A,D,threshold);
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
          child, child_dict = C45(data, A, threshold, node, dict_) 
          edge = Edge(option_labels[i], child)

          edge_dict = {"edge": {"value": option_labels[i], "node": child_dict}}
          node.edges.append(edge)
          node_dict["node"]['edges'].append(edge_dict)

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

def selectSplittingAttribute(A,D,threshold):
  gain = []
  print("A", A)
  p0, option_labels = first_entropy(D, D.columns[-1])
  for i in range(len(A)):
    pA = entropy(D, A[i], D.columns[-1])
    gain.append(p0 - pA) 
  best = max(gain)
  print("best", best)
  print("threshold", threshold)
  # print("Best: ", best)
  if best > threshold:
    x = A[gain.index(max(gain))]
    return x
  else:
    return None


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


main()