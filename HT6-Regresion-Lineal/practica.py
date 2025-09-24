from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
wine= load_wine()  

x = wine.data
y = wine.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=MLPClassifier(hidden_layer_sizes=(100,150), activation='relu',solver='adam',max_iter=1000,random_state=42)

model.fit(x,y)
y_pred=model.predict(x_test)

print(y_test)
print(y_pred)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

indiv=x_test[0].reshape(1,-1)
print(indiv)