
import pandas as pd
import numpy as np
import random 

df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 1]]),
                   columns=['a', 'b', 'c'])

print(df)
for i in df.columns:
    df.drop(df.loc[df[i] == 1].index, inplace=True)

print(df)

here = [0] * 3
print(here)
for i in range(3):
    here[i] += i

print(here[0])

x = np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])

x = np.append(x, [0 for i in range(3)])

# print(x)
# print([0 for _ in range(3)])
# i = np.array([1,2,3])
# np.insert([0 for _ in range(1)], 0, i, axis=0)

# here = np.array([2,2])
# div = 2

# final = here/div

# print(final)

from collections import Counter
d1 = Counter({'a': 2, 'b': 9, 'c': 8, 'd': 7})
print(d1)
d2 = Counter({'a': 2, 'b': 2, 'c': 3, 'e': 2})
print(d2)
d3 = d1 - d2
print(d3)

values = [1,2,3]

here = {x:1 for x in values}

print(here)