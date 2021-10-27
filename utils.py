import sys 
import pandas as pd 
import math 
import json 
import numpy as np
import random

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
    A = list(D)
    return n, D, A

def is_numeric(D): 
  numeric = {}
  for col in D.columns:
    if int(D.at[0,col]) == 0: 

      numeric[col] = True 
    else: 
      numeric[col] = False 

  return numeric

def find_result(attributes, list_, dict_): 
  result = ""
  if "node" in  dict_.keys():
    node = dict_["node"]
    result = find_result(attributes, list_,node)
  elif "leaf" in  dict_.keys():
    return dict_['leaf']['decision']
    
  elif "edges" in  dict_.keys():
    edges = dict_["edges"]

    first_edge = edges[0]['edge']
    if "direction" in first_edge:
      
      edge_value = first_edge["value"]
      second_edge = edges[1]['edge']
      attribute = dict_["var"]


      passed_in_value = float(list_[attributes.index(attribute)])


      if passed_in_value <= edge_value:
        if first_edge["direction"] == "le": 
          result =  find_result(attributes, list_, first_edge["node"])
        else: 
          result =  find_result(attributes, list_,second_edge["node"])
          
      else: 
        if first_edge["direction"] == "gt": 
          result =  find_result(attributes, list_, first_edge["node"])
        else: 
          result =  find_result(attributes, list_,second_edge["node"])

    else: 
      for edge in edges: 
        if edge['edge']['value'] in list_: 
          edge_of_interest = edge['edge']
          if "leaf" in edge_of_interest.keys():
            return  edge['edge']['leaf']['decision']
          else: 
            result = find_result(attributes, list_,edge_of_interest["node"])

      
  return result 

def zero_matrix(result):
  zeros = []
  for i in range(len(result)):
    zeros.append([0]*len(result))

  actual = [] 
  classified = []
  for i in result:
    actual.append("A "+ str(i))
    classified.append("C "+ str(i))

  data = pd.DataFrame(data = zeros, index = actual, columns = classified)

  return data

def matrix(a,b, result):
  data = zero_matrix(result)
  for i, j in zip(a, b):
    data["C "+ str(i)]["A "+ str(j)] = data["C "+ str(i) ]["A "+ str(j)] + 1
  return data

def classifier(D, raw, k, Forest):
  
  json_filename = "json_out"
  f = open(json_filename, "w")
  json.dump(raw, f)
  f.close()

  objects = list(D[D.columns[-1]])
  object_type = list(set(objects))
  attributes = list(D.iloc[:1])


  overall = [[] for _ in range(3)]
  overall_r = []
  o_data = []
  records = len(D)

  for index, row in D.iterrows():
    passed_row = row.tolist()
    if len(attributes) != len(passed_row): 
      print("ERROR")
      sys.exit()

    result = find_result(attributes, passed_row, raw) 
    result1 = knn(D, k, index)
    result2 =  RFClassify(Forest, passed_row)

    overall[0].append(result)
    overall[1].append(result1)
    overall[2].append(result2)
  
  for i in range(3):
    x = D[D.columns[-1]].to_list()
    data = {'Predicted': overall[i], "Real": x}
    df = pd.DataFrame(data)
    print(df)

    a = df['Predicted'].to_list()
    b = df['Real'].to_list()

    result = df['Real'].value_counts().index.to_list()
    result2 = df['Predicted'].value_counts().index.to_list()

    for i in result2:
      if (i not in result):
        result.append(i)

    data = matrix(a,b, result)
    o_data.append(data)
    overall_r.append(result)

  
  return records, o_data, overall_r

def classifier2(D, raw):
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
    result = find_result(attributes, passed_row, raw) #C45 implimentation 
    overall.append(result)


  x = D[D.columns[-1]].to_list()
  data = {'Predicted': overall, "Real": x}
  df = pd.DataFrame(data)

  a = df['Predicted'].to_list()
  b = df['Real'].to_list()

  result = df['Real'].value_counts().index.to_list()

  data = matrix(a,b, result)
  o_data.append(data)
  overall_r.append(result)

  return records, data, result 

def cross_validation2(is_numeric, D, A, n, threshold):
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
  # n = amount of folds 
  while i <= n:
    num = round(overall/n)
    if (num > o_eval or (i - n) == 0):
      num = o_eval
    o_eval = o_eval - num
    end = total_amt + num - 1
    if (total_amt+num) > overall:
      end = overall

    send_rec = D.loc[total_amt:end].reset_index(drop=True)
    # send rec = amount in the data set, gets smaller every time 3/3 2/3 1/3

    if (n == 1):
      send_C45 = send_rec
    else:
      send_C45 = D.drop(range(total_amt,end)).reset_index(drop=True)

    before_node = Normal()
    dict_ = {}
    Attributes = A.copy()

    node, final_dict = C45(is_numeric, send_C45, Attributes, threshold, before_node, dict_)  
    # print(final_dict)
    i+=1

    total, data, result = classifier2(D, final_dict)
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


  # print("Overall Matrix of: C45")

  overall_matrix = zero_matrix(result)
  for j in matrix_fin:
    overall_matrix = overall_matrix.add(j, fill_value=0)
  print(overall_matrix)

  print("Totals: ")
  #print("Total Number of records classified: ", total_amt)
  print("Overall Accuracy: ", correct/total_amt)
  print("Individual accuracies: ", accuracy)
  print("Average accuracy: ", sum(accuracy)/len(accuracy))

def cross_validation(is_numeric, D, A, n, threshold, m, k_rt, N, k_knn):
  D = D.sample(frac = 1).reset_index(drop=True)
  overall = len(D)
  o_eval = len(D)
  correct = [0]*3 
  total_amt = 0
  accuracy = [[] for _ in range(3)]

  if(n == 0):
    n = 1
  elif(n == -1):
    n = len(D)

  i = 1 
  # n = amount of folds 
  while i <= n:
    num = round(overall/n)
    if (num > o_eval or (i - n) == 0):
      num = o_eval
    o_eval = o_eval - num
    end = total_amt + num - 1
    if (total_amt+num) > overall:
      end = overall

    send_rec = D.loc[total_amt:end].reset_index(drop=True)
    # send rec = amount in the data set, gets smaller every time 3/3 2/3 1/3
    if (n == 1):
      send_C45 = send_rec
    else:
      send_C45 = D.drop(range(total_amt,end)).reset_index(drop=True)

    before_node = Normal()
    dict_ = {}
    Attributes = A.copy()

    node, final_dict = C45(is_numeric, send_C45, Attributes, threshold, before_node, dict_)  

    i +=1

  forest = randomForest(D, A, N, m, k_rt, is_numeric)

  total, data, result = classifier(D, final_dict, k_knn, forest)

  names = ["C45", "KNN", "Random Forest"]
  matrix_fin = []
  for i in range(len(data)):
    o_matrix = []

    lst = data[i].values.tolist()

    o_matrix.append(data[i])
    total = data[i].values.sum()

    T = 0
    for j in range(len(lst)):
      T = T + lst[j][j]

    
    correct[i] += T
    accuracy[i].append(T/total)
    matrix_fin.append(o_matrix)
  total_amt += total

  for i in range(3):
    print()
    print("Overall Matrix of: ", names[i])

    overall_matrix = zero_matrix(result[i])
    for j in matrix_fin[i]:
      overall_matrix = overall_matrix.add(j, fill_value=0)
    print(overall_matrix)

    print()
    print("Totals: ")
    #print("Total Number of records classified: ", total_amt)
    print("Overall Accuracy: ", correct[i]/total_amt)
    print("Individual accuracies: ", accuracy[i])
    print("Average accuracy: ", sum(accuracy[i])/len(accuracy[i]))

def json_file_to_dict(json_file): 
  with open(json_file) as f:
    data = json.load(f)
    return data 

def knn(D, k, index):
  dist = [0] * len(D)
  actual = D[D.columns[-1]].to_list()
  D = D.iloc[:,:-1] 
  D = D.apply(pd.to_numeric, errors='coerce')

  normalized_df=(D-D.min())/(D.max()-D.min())

  p_classify = normalized_df.iloc[index]

  p_classify = [float(x) for x in p_classify]
  p_classify = np.array(p_classify)

  for index, row in normalized_df.iterrows():
    new_row = np.array(row.tolist())

    dist[index] = dist[index] + np.linalg.norm(p_classify-new_row)
  

  data = {'Distances': dist, "Predictions": actual}
  df = pd.DataFrame(data)


  df = df.sort_values(by=['Distances']).reset_index(drop=True)

  df = df[1:k+1]

  prediction = df["Predictions"].value_counts().index.max()

  return prediction

def C45(is_numeric, D, A, threshold, node, dict_):
  # print("C45")
  best = []
  bol, attr = check_home(D, A)

  if (bol == True): 
    leaf = Leaf(1,attr)
    if type(attr) == tuple: 
      a = attr[0]
    else: 
      a = attr 
    leaf_dict = {"leaf": {"decision": a, "p":1}}
    # print("leaf_dict 1")
    return leaf , leaf_dict   

  elif not A:
    c, p = find_most_frequent_label(D)
    leaf = Leaf(1,c)
    leaf_dict = {"leaf": {"decision": c, "p": p}}
    # print("leaf_dict 2")
    return leaf , leaf_dict 
    
  else:
    Ag, x = selectSplittingAttribute(is_numeric, A,D,threshold)
    # print("Ag", Ag)
    if Ag == None:
      c, p = find_most_frequent_label(D)
      leaf = Leaf(1,c)
      leaf_dict = {"leaf": {"decision": c, "p": p}}
      # print("leaf_dict 3")
      return leaf, leaf_dict 

    else:
      try:
        A.remove(Ag)
      except:
        node = Normal(Ag)
        node_dict = {"node": { "var": x,"edges": []}}
        option_labels = ['le', 'gt']
        for i in range(len(option_labels)):
          if option_labels[i] == 'le':
            data = D[D[x] <= Ag]
          else:
            data = D[D[x] > Ag]
        
          if (data.empty):
            # NEW GHOST PATHS CODE
            continue 
            # c, p = find_most_frequent_label(D)
            # leaf = Leaf(1,c)
            # leaf_dict = {"leaf": {"decision": c, "p": p}}
            # return leaf, leaf_dict  
            #########################

          child, child_dict = C45(is_numeric, data, A, threshold, node, dict_) 
          edge = Edge(option_labels[i], child)
          edge_dict = {"edge": {"value": float(Ag), "direction": option_labels[i], "node": child_dict}}
          node.edges.append(edge)
          node_dict["node"]['edges'].append(edge_dict)

      else:
        node = Normal(Ag)
        node_dict = {"node": { "var": Ag,"edges": []}}
        
        labels = list(D[Ag])
        option_labels = list(set(labels))

        for i in range(len(option_labels)):
          try:
            data = D.loc[(D[Ag] == option_labels[i])] 
          except:
            continue
          else:
            child, child_dict = C45(is_numeric, data, A, threshold, node, dict_) 
            
            edge = Edge(option_labels[i], child)

            edge_dict = {"edge": {"value": option_labels[i], "node": child_dict}}
            node.edges.append(edge)
            node_dict["node"]['edges'].append(edge_dict)
            

  # print("node_dict", node_dict)
  return node, node_dict

def most_frequent(list_):
  return max(set(list_), key = list_.count)

def RFClassify(Forest, x):
  prediction = []
  for f in Forest: 
    prediction.append( find_result(f.A, x, f.d,))
  
  return most_frequent(prediction)
  
def randomForest(D, A, N, m, k, is_numeric):
  forest = []
  m = int(round(m*len(A)))
  k = int(round(k*len(D)))

  for i in range(0,N): 
    AttList = random.sample(A, m)
    Data =  D.sample(n = k)
    f = Forest(AttList, Data, 0, is_numeric)
    forest.append(f)
    i+=1
    
  return forest

class Forest: 
  def __init__(self, A, D,threshold, is_numeric): 
    self.A = A
    self.D = D
    self.threshold = threshold
    node = Normal()
    d = {}
    T, d = C45(is_numeric, D, A,threshold,node,d)
    self.d = d

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
      self.node = node  
    def __repr__(self):
      ret = "{} {}".format( self.label,  self.node.__repr__())
      return ret 

def find_most_frequent_label(D): 
  df = D.apply(pd.Series.value_counts)
  max_column = df.columns[-1]
  label = df[max_column].idxmax()
  percentage = df[max_column].max()/df[max_column].sum()
  return label, percentage

def selectSplittingAttribute(is_numeric, A,D,threshold):
  gain = []
  check = []
  p0, option_labels = first_entropy(D, D.columns[-1])
  pA = 0
  for i in range(len(A)):
    if is_numeric[A[i]] == True:

      x, pA = findBestSplit(D.columns[-1], A[i], D) #returns best splitting attribute and calculated pA
      print("bestsplit ", x, pA)
      sys.exit()
    else:
      pA = entropy(D, A[i], D.columns[-1])
      x = 'category'
    
    check.append(x)
    gain.append(p0 - pA) 
  
  best = max(gain)
  # print("best", best)
  # print("threshold", threshold)
  if best > threshold:
    mx = A[gain.index(best)]
    x = check[gain.index(best)]
    if x != 'category':
      return x, mx
    else:
      return mx, x
  else:
    return None, None
  
def compute_e(overall_r_1, overall_r_2, option_labels):
  e_left = 0
  e_right = 0 
  for i in range(len(option_labels)):
      if (sum(overall_r_1) == 0):
        left = 0 
      else:
        left = overall_r_1[i]/sum(overall_r_1)

      if (sum(overall_r_2) == 0):
        right = 0 
      else:
        right = overall_r_2[i]/sum(overall_r_2)
      
      e_left = e_left + log(left)
      e_right = e_right + log(right)
  return e_left, e_right

def entropy_split(e_left, e_right, overall_r_1, overall_r_2):
  return (sum(overall_r_1)/(sum(overall_r_1) + sum(overall_r_2)))*e_left + (sum(overall_r_2)/(sum(overall_r_1) + sum(overall_r_2)))*e_right

def findBestSplit(A, a, data):
  print("\nFind Best Split")
  print("A", A)
  print("a",a)
  print("D", data)


  ageValues = np.sort(data[a].unique())


  myAtt = a
  splits = [data[data[myAtt]<= i] for i in ageValues]
  print("splits\n", splits)
  print(type(splits))
  sizes = np.array([len(splits[i]) for i in range(len(ageValues))])
  print("sizes\n", sizes, len(sizes))
  counts = np.array([np.array(x[A].value_counts()) for x in splits ])
  print("counts\n", counts, len(counts))


  fullDistribution = np.array(data[A].value_counts())
  print("full Distribution", fullDistribution)
  print("n",n)

  f = np.array([c/l for c,l in zip(counts, sizes)])
  print("f", f)
  return None, None
  # rightSide = fullDistribution-counts
  # print("rightSide", rightSide)


  # g = np.array([c/l for c,l in zip(rightSide, n-sizes)])
  # print("g",g)

  # split_entropies = sizes/n * (-np.sum(f*np.log2(f), axis = 1))  +   sizes/(n-sizes) * -np.sum(g*np.log2(g), axis=1)
  # print("Split entropies", split_entropies)
  # return ageValues[np.argmin(split_entropies[:-1])], np.argmin(split_entropies[:-1])

  # END 








  # entropy1, option_labels = first_entropy(D, A)
  # if (a == A):
  #   return entropy1

  # pd.to_numeric(D.loc[:, a])
  # labels2 = D[a].value_counts().index.tolist()
  # labels2.sort()
  # option_labels2 = labels2

  # track_lab = [] #label tracker so we can send back the one that has the best entropy 
  # track_entro = []

  # for i in range (len(option_labels2)): 
  #   overall_r_1 = []
  #   overall_r_2 = []
  #   for j in range (len(option_labels)): 
  #     label = option_labels2[i]
  #     total = int((D[A] == option_labels[j]).sum())
  #     ratio_1 = len(D.loc[(D[a] <= label) & (D[A] == option_labels[j])].index)
  #     ratio_2 = total - ratio_1
  #     overall_r_1.append(ratio_1)
  #     overall_r_2.append(ratio_2)

  #   e_left, e_right = compute_e(overall_r_1, overall_r_2, option_labels)
  #   e_overall = entropy_split(e_left, e_right, overall_r_1, overall_r_2)
  #   track_entro.append(e_overall)
    
  # best = min(track_entro)
  # o_label = option_labels2[track_entro.index(best)]
  # mx_v = best
  # return o_label, mx_v 

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










