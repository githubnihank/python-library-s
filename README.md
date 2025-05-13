# python-library-s
Data science using pyhton library 's

PANDAS library 

import pandas as pd

data = {'name':['nihank'],'age':[25]}

df = pd.DataFrame(data)

print(df)

OUTPUT :   name  age
0  nihank   25

NUMPY library

import numpy as np

array =  np.array([[1,2],[3,4]])

result = np.dot(array,array)

print(result)

 NumPy array [[1, 2], [3, 4]] with itself. In the case of a 2x2 matrix, the dot product is calculated as follows: So, for this case:

Result[0][0] = (1*1) + (2*3) = 1 + 6 = 7
Result[0][1] = (1*2) + (2*4) = 2 + 8 = 10
Result[1][0] = (3*1) + (4*3) = 3 + 12 = 15
Result[1][1] = (3*2) + (4*4) = 6 + 16 = 22

OUTPUT : [[ 7 10]
 [15 22]]

matplotlib library

![image](https://github.com/user-attachments/assets/ea848307-bfba-4048-942a-c1123814504b)

import matplotlib.pyplot as plt

x=[1,2,3,4,]

y=[1,4,9,]

plt.plot(x,y)

plt.show()

SEABORN LIBRARY

![image](https://github.com/user-attachments/assets/d2c83a70-be40-45f8-b52c-f8b55f8e9a57)

import seaborn as sns

import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

sns.pairplot(data)

plt.show()
