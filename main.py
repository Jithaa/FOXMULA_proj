import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
data=pd.read_csv("train.csv").values

clf=DecisionTreeClassifier()
xtrain=data[0:30000,1:]
xlabel=data[0:30000,:1]
clf.fit(xtrain,xlabel)
ytest=data[30000:,1:]
ylabel=data[30000:,0]

for i in range(0,5):
    t=ytest[i]
    t.shape=(28,28)
    plt.imshow(255-t,cmap='gray')
    print(clf.predict([ytest[i]]))
    plt.show()