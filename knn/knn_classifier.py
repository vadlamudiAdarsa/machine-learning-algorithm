import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
x=[0.7,0.8,1.1,0.7,1.1,3,3.7,4.1,3.7,4.1,4.3,4.3,3.1,3.1,3.7,3.4,3.9,3.9]
y=[0.7,0.8,0.7,1.1,1.1,2,2.7,2.7,3.1,3.1,2.7,3.1,0.3,0.6,0.4,0.6,0.9,0.6]
classes=[1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3]
plt.scatter(x,y,c=classes)
plt.text(2.5,3,s="Before classificaiton")
plt.show()
data=list(zip(x,y))
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(data,classes)
new_x=2.1
new_y=0.7
new_point=[(new_x,new_y)]
prediction=knn.predict(new_point)
print("Testng patten belongs to the class:",prediction)
plt.scatter(x+[new_x],y+[new_y],c=classes+[prediction[0]])
plt.text(2.5,3,s="Before classificaiton")
plt.text(x=new_x-0.5,y=new_y-0.4,s="New pattern:"+str(prediction))
plt.show()
