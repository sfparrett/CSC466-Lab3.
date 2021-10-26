
import pandas as pd
import numpy as np

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