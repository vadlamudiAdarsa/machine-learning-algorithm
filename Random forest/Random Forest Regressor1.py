#implementation of Random Forest Regressor
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
df=pd.read_csv("example4.csv")
#mapping of string data to int
a={'<=30':0,'31-40':1,'>40':2}
df['age']=df['age'].map(a)
i={'high':0,'medium':1,'low':2}
df['income']=df['income'].map(i)
s={'no':0,'yes':1}
df['student']=df['student'].map(s)
c={'fair':0,'excellent':1}
df['credit_rating']=df['credit_rating'].map(c)
cs={'no':0,'yes':1}
df['class:buys_computer']=df['class:buys_computer'].map(cs)
#specifying Features and classes
features=['age','income','student','credit_rating']
x=df[features]
y=df['class:buys_computer']
#classification based on random Forest
rr=RandomForestRegressor()
model=rr.fit(x,y)

#testing pattern
test_age=a['>40']
test_income=i['high']
test_student=s['no']
test_credit=c['fair']
test_pattern=[test_age,test_income,test_student,test_credit]
p=rr.predict([test_pattern])
print("Testing pattern is assigned to the class:",p,"Based on random Forest Regressor")

#pick one tree from the forest ex:index=0
tree=rr.estimators_[0]
#plot the decision tree
plt.figure.figsize=[20,10]
plot_tree(tree,feature_names=df.columns.tolist(),filled=True,rounded=True, fontsize=10)
plt.title("Decision Tree from Random Forest 0")
plt.show() 