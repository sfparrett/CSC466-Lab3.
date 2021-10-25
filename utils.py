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

def find_result(attributes, list_, dict_): 
  print("\n")
  print("dict_", dict_)

  result = ""
  if "node" in  dict_.keys():
    node = dict_["node"]
    # print("node ", node)
    result = find_result(attributes, list_,node)
  elif "leaf" in  dict_.keys():
    print("leaf ", dict_['leaf']['decision'])
    return dict_['leaf']['decision']
    
  elif "edges" in  dict_.keys():
    print("numerical edges")
    edges = dict_["edges"]

    first_edge = edges[0]['edge']
    if "direction" in first_edge:
      
      edge_value = first_edge["value"]
      print("edge value ", edge_value, type(edge_value))
      second_edge = edges[1]['edge']
      attribute = dict_["var"]
      print("attribute ", attribute, type(attribute))

      passed_in_value = float(list_[attributes.index(attribute)])
      print("passed in value ", passed_in_value, type(passed_in_value))
      print("first value\n", first_edge)
      print("second value\n", second_edge)

      if passed_in_value <= edge_value:
        if first_edge["direction"] == "le": 
          result =  find_result(attributes, list_, first_edge["node"])
        else: 
          result =  find_result(attributes, list_,second_edge["node"])
          
      else: # greater than 
        if first_edge["direction"] == "gt": 
          result =  find_result(attributes, list_, first_edge["node"])
        else: 
          result =  find_result(attributes, list_,second_edge["node"])

    else: 
      # print("categorical edges")
      for edge in edges: 
        if edge['edge']['value'] in list_: 
          edge_of_interest = edge['edge']
          if "leaf" in edge_of_interest.keys():
            return  edge['edge']['leaf']['decision']
          else: 
            result = find_result(attributes, list_,edge_of_interest["node"])

      
  # print("RESULT {}".format(result))
  return result 

def zero_matrix(result):
  zeros = []
  print(result)
  for i in range(len(result)):
    zeros.append([0]*len(result))

  actual = [] 
  classified = []
  for i in result:
    print("i ", i)
    print(type(i))
    actual.append("A "+ str(i))
    classified.append("C "+ str(i))
    # actual.append("Actual "+ str(i))
    # classified.append("Classified "+ str(i))
  
  data = pd.DataFrame(data = zeros, index = actual, columns = classified)
  pd.set_option('display.max_columns', None)
  print("Zero Matrix: ")
  print(data)

  return data


def matrix(a,b, result):
  data = zero_matrix(result)

  for i, j in zip(a, b):
    data["C "+ str(i)]["A "+ str(j)] = data["C "+ str(i) ]["A "+ str(j)] + 1
    #data["Classified "+ str(i)]["Actual "+ str(j)] = data["Classified "+ str(i) ]["Actual "+ str(j)] + 1


  return data

def classifier(D, raw):
  
  json_filename = "json_out"
  f = open(json_filename, "w")
  json.dump(raw, f)
  f.close()

  objects = list(D[D.columns[-1]])
  object_type = list(set(objects))
  attributes = list(D.iloc[:1])
  print("UNIQUE VALUES\n", D[attributes[-1]].unique()) # [6. 7. 5. 4. 8. 3.]

  overall = []
  records = len(D)
  print("RESULTS: ")
  for index, row in D.iterrows():
    passed_row = row.tolist()
    if len(attributes) != len(passed_row): 
      print("ERROR")
      sys.exit()
    #passed_row = [ 6.9,1.09,0.06,2.1,0.061,12,31,0.9948,3.51,0.43,11.4,4] #4
    #passed_row = [7.1,1.5,0.01,5.7,0.082,3,14,0.99808,3.4,0.52,11.2,3] #3
    result = find_result(attributes, passed_row, raw)
    print("result ", result)
    # sys.exit()
    overall.append(result)
    

  

  x = D[D.columns[-1]].to_list()
  data = {'Predicted': overall, "Real": x}
  df = pd.DataFrame(data)
  print(df)

  a = df['Predicted'].to_list()
  b = df['Real'].to_list()

  result = df['Real'].value_counts().index.to_list()
  result2 = df['Predicted'].value_counts().index.to_list()

  print("Real vs. Predicted")
  print(result)
  print(result2)
  print()
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

    print(send_rec)

    if (n == 1):
      send_C45 = send_rec
    else:
      send_C45 = D.drop(range(total_amt,end)).reset_index(drop=True)

    before_node = Normal()
    dict_ = {}
    Attributes = A.copy()

    node, final_dict = C45(is_numeric, send_C45, Attributes, threshold, before_node, dict_)

    print()
    print("Final: ")
    print(final_dict)

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
    c, p = find_most_frequent_label(D)
    leaf = Leaf(1,c)
    leaf_dict = {"leaf": {"decision": c, "p": p}}
    return leaf , leaf_dict 
    
  else:
    Ag, x = selectSplittingAttribute(is_numeric, A,D,threshold)
    if Ag == None:
      c, p = find_most_frequent_label(D)
      leaf = Leaf(1,c)
      leaf_dict = {"leaf": {"decision": c, "p": p}}
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
            #continue
            # NEW GHOST PATHS CODE 
            c, p = find_most_frequent_label(D)
            leaf = Leaf(1,c)
            leaf_dict = {"leaf": {"decision": c, "p": p}}
            return leaf, leaf_dict  
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
  check = []
  p0, option_labels = first_entropy(D, D.columns[-1])
  #print("p0: ", p0)
  pA = 0
  #print("Is_numeric: ", is_numeric)
  #print("A: ", A)
  for i in range(len(A)):
    if is_numeric[A[i]] == True:
      #print()
      x, pA = findBestSplit(D.columns[-1], A[i], D) #returns best splitting attribute and calculated pA
      #entropy x <= 3 < x 
      #print("X: ", x)
      #print("pA: ", pA)
      #print()
    else:
      pA = entropy(D, A[i], D.columns[-1])
      x = 'category'
    
    check.append(x)
    gain.append(p0 - pA) 
  
  #print("Check: ", check)
  #print("Gain: ", gain)
  
  best = max(gain)
  if best > threshold:
    mx = A[gain.index(best)]
    #print("mx", mx)
    x = check[gain.index(best)]
    #print("x", x)
    if x != 'category':
      return x, mx
    else:
      return mx, x
  else:
    return None, None
  

  
  

  # for i in range(len(A)):
  #   if (D[A[i]].iloc[0] == str(D[A[i]].iloc[0])):  
  #     if D[A[i]].iloc[0].isnumeric(): 
  #       D[A[i]] = pd.to_numeric(D[A[i]])
  #       x = findBestSplit(A[i], D)
  #       sys.exit()
  #       pA = entropy(D, A[i], D.columns[-1])
  #     else:
  #       pA = entropy(D, A[i], D.columns[-1])
  #   elif D[A[i]].iloc[0].is_integer():
  #     x = findBestSplit(A[i], D)
  #     sys.exit()
  #     pA = entropy(D, A[i], D.columns[-1])
  #   gain.append(p0 - pA) 
  # best = max(gain)
  # if best > threshold:
  #   x = A[gain.index(max(gain))]
  #   return x
  # else:
  #   return None

def compute_e(overall_r_1, overall_r_2, option_labels):
  e_left = 0
  e_right = 0 
  for i in range(len(option_labels)):
      #print("i: ", i)
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


def findBestSplit(A, a, D):
  entropy1, option_labels = first_entropy(D, A)
  if (a == A):
    return entropy1

  pd.to_numeric(D.loc[:, a])
  #print()
  #print("FIND BEST SPLIT")
  #print()
  #print(D[a])
  labels2 = D[a].value_counts().index.tolist()
  labels2.sort()
  option_labels2 = labels2
  #print("OL2: ", option_labels2)

  track_lab = [] #label tracker so we can send back the one that has the best entropy 
  track_entro = []

  #print("First Entropy", entropy1)
  #print("A: ", a)
  #print(D.sort_values(by=[a]))
  for i in range (len(option_labels2)): 

    #print()
    #print("Label ", option_labels2[i])
    overall_r_1 = []
    overall_r_2 = []
    for j in range (len(option_labels)): 
      label = option_labels2[i]
      total = int((D[A] == option_labels[j]).sum())
      ratio_1 = len(D.loc[(D[a] <= label) & (D[A] == option_labels[j]) ].index)
      ratio_2 = total - ratio_1
      overall_r_1.append(ratio_1)
      overall_r_2.append(ratio_2)
      #print("ratio_1: ", ratio_1, "ratio_2: ", ratio_2)
      # label = option_labels2[i]
      # total = int((D[a] == label).sum())
      # ratio_1 = len(D.loc[(D[a] == label) & (D[A] == option_labels[j]) ].index) / total
    
    #print("left: ", overall_r_1)
    #print("right: ", overall_r_2)
    #compute e 
    e_left, e_right = compute_e(overall_r_1, overall_r_2, option_labels)
    #print("e_left: ", e_left, "e_right: ", e_right)

    e_overall = entropy_split(e_left, e_right, overall_r_1, overall_r_2)
    #print("e_overall: ", e_overall)

    track_entro.append(e_overall)
    
  best = min(track_entro)
  o_label = option_labels2[track_entro.index(best)]
  #print(best)
  #print(o_label)
  mx_v = best
  return o_label, mx_v 



  # #  Questions: 
  # # 1.) What does ascendin g order look like for D
  # # 2.) what is alpha 
  # print("A", a)
  # counts = [[]]
  # Gain = []
  # alpha = []
  # p0 = first_entropy(D, a) #we might need to change this 
  # D = D.sort_values(by=[a], ascending = True) # ascending = True or False?
  # D= D.reset_index(drop=True)
  # for i in range(1, len(D)): 
  #   d = D.iloc[i-1].to_list()
  #   alpha.append(d[0])
  #   print("Alpha: ", alpha)
  #   exit()
  #    # go into data point , check this   
  #   for j in range(len(k)):
  #     continue 
  # for l in range(1, len(D)):
  #   Gain[l] = p0 - entropy(D, a, counts) #some "magic" has been hidden inside the call of the entropy(D,Ai, (counts_1[l],...,counts_k[l]))  
  #                                        #function (we are passing it the distribution of the class labels on the "left" side of the split, 
  #                                        #the function needs to construct the distribution of labels on the "right" side of the split from the data passed into it)..
  # pass 

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
  #print("Check_hom ", check_home) 
  d = D[[D.columns[-1]]]
  #print("d ", d)
  results = d.value_counts().index.to_list()
  #print("results ", results)
  if len(results) > 1: 
    return False, False 
  else: 
    return True, results[0]

