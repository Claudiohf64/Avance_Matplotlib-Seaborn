#tareA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA
#TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA
#TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA
#TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA
#TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA  
#TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA
#TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA
#TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA
#TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA
#TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA
#TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA TAREA
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
california = fetch_california_housing()
x = california.data
y = california.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("R^2 Score:",r2)
print(california.DESCR)