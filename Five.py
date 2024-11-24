# 5. Build a classification model using Decision Tree algorithm on iris dataset.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
iris = sns.load_dataset('iris')
iris.head()

x=iris.iloc[:,:-1]
y=iris.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

treemodel = DecisionTreeClassifier()
treemodel.fit(x_train, y_train)
y_pred = treemodel.predict(x_test)
plt.figure(figsize=(20,30))
tree.plot_tree(treemodel, filled=True)
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: ')
print(cm)
