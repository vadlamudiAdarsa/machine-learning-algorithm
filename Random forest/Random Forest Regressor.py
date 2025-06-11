#implementation of Random Forest Regressor
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
df=pd.read_csv("example3.csv")

#specifying Features and classes
features=['feature1','feature2']
x=df[features]
y=df['target']
#classification based on random Forest
rr=RandomForestRegressor()
model=rr.fit(x,y)

#testing pattern
test_pattern=[0.5,0.7]
p=rr.predict([test_pattern])
print("Testing pattern is assigned to the class:",p,"Based on random Forest Regressor")

#pick one tree from the forest ex:index=0
tree=rr.estimators_[0]
#plot the decision tree
plt.figure.figsize=[20,10]
plot_tree(tree,feature_names=df.columns.tolist(),filled=True,rounded=True, fontsize=10)
plt.title("Decision Tree from Random Forest 0")
plt.show() 