
#apply knn algorithm for regression
import pandas as pd
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor

#training patterns
df=pd.read_csv("Example2.csv")
x=df[['length','width']]
y=df['cost']
#applying linear  regression on training pattern
r=linear_model.LinearRegression()
r.fit(x.values,y.values)
#prediction based on linear regression
y_p=r.predict([[7,8]])
print("Predicted value using linear regression:",y_p)
#applying knn regression algorithm on training pattern
knn_r=KNeighborsRegressor(n_neighbors=5)
knn_r.fit(x.values,y.values)
#prediction based on knn regression
y_p_r=knn_r.predict([[7,8]])
print("Predicted value based on knn regression:",y_p_r)
