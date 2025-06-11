#implementation of decision tree regressor
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

df=pd.read_csv("Example5.csv")

#specifying features and classes
features =['feature1','feature2']
x=df[features]
y=df['Target']

#Regressor based on Decision tree
dtree=DecisionTreeRegressor()
model=dtree.fit(x.values,y.values)
#prediction based on Decision tree
p=dtree.predict([[0.3,0.4]])
print("Predicted value using DecisionTreeRegressor is:",p)

#Text representation of Tree
#text=tree.export_text(dtree)
#print(text)

#Visualizing decision tree
plot_tree(model,feature_names=features,fontsize=10)
plt.figure(figsize=(8,8))
plt.title("Decision tree structure")
plt.show()