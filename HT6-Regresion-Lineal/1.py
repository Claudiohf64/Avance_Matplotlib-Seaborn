#importamos MLPClassifier de sklearn para crear una red neuronal
#importamos también algunas utilidades para evaluar el modelo como accuracy_score y classification_report
#importamos train_test_split para dividir el dataset en conjunto de entrenamiento y prueba
#importamos load_iris para cargar el dataset de iris
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#cargamos el dataset de iris
iris = load_iris()
x = iris.data
y = iris.target

#dividimos el dataset en conjunto de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#el tamaño de la capa oculta es (10, 8) lo hacemos con hidden_layer_sizes
#usamos max_iter para definir el número máximo de iteraciones
model = MLPClassifier(hidden_layer_sizes=(10, 8), activation='relu', solver='adam', max_iter=1000, random_state=42)

#entrenamos el modelo
model.fit(x_train, y_train)

#hacemos predicciones
y_pred = model.predict(x_test)

#evaluamos el modelo
accuracy = accuracy_score(y_test, y_pred)

#hacemos un reporte de clasificación
report = classification_report(y_test, y_pred)

#imprimimos los resultados de la evaluación
print(f"Accuracy: {accuracy}")

#imprimimos la matriz de confusión
print("Classification Report:")
print(report)

#imprimimos las etiquetas reales y las predicciones
print ("Etiquetas reales:", y_test)
print("Predicciones:", y_pred)