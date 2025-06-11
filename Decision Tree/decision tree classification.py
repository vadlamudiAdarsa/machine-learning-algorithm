#implementation of decision tree classification
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df=pd.read_csv("Example3.csv")

#specifying features and classes
features =['Feature1','Feature2']
x=df[features]
y=df['Class']

#classification based on Decision tree
dtree=DecisionTreeClassifier()
model=dtree.fit(x,y)
plt.figure(figsize=(8,8))
tree.plot_tree(model,feature_names=features)

#Text representation of Tree
text=tree.export_text(dtree)
print(text)
plt.show()