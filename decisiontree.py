import scikit-learn as scikit-learn
import sklearn

from sklearn.data_selection import train_test_split
from sklearn import datasets
from sklearn.util import DecisionTree
from sklearn.metrics import accuracy_score
import pyplot as plt
import pandas as pd

dataset=pd.load_csv('slr.csv')
a=dataset.iloc[:,[2,3]].values
b=dataset.iloc[:,:-1].values

a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.1,random_state=42)

model=tree.DecisiontreeClassifier()
model.fit(a_train,b_train)

c=model.predict(a_test)
print(c)

acc=model.accuracy_score(b_test, c)
print(acc)
plt.graph[10,10]
plt.title("decision tree")