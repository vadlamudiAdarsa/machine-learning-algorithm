#implementation of Random Forest Classifier
import pandas as pd
from sklearn import metrics
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv("example2.csv")
do={'sunny':0,'overcast':1,'rainy':2}
df['outlook']=df['outlook'].map(do)
dc={'NP':0,'P':1}
df['Class']=df['Class'].map(dc)
#specifying Features and classes
features=['outlook','temp(F)','humidity']
x=df[features]
y=df['Class']
#classification based on random Forest

rc=RandomForestClassifier()
model=rc.fit(x,y)

#testing pattern
new_x='rainy'
new_y=65
new_z=75
new_x_mapped = do[new_x]
new_point=[[new_x_mapped,new_y,new_z]]
p=rc.predict(new_point)
print("Testing pattern is assigned to the class:",p,"Based on random Forest Classifier")
#using metrics module for accuracy calculations
print("Accuracy of the model:",metrics.accuracy_score([new_y],p))

#pick one tree from the forest ex:index=0
tree=rc.estimators_[0]
#plot the decision tree
plt.figure.figsize=[20,10]
plot_tree(tree,feature_names=df.columns.tolist(),filled=True,rounded=True, fontsize=10)
plt.title("Decision Tree from Random Forest 0")
plt.show() 