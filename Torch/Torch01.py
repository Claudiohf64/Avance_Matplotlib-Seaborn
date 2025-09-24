from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Cargar Fashion MNIST
fashion = fetch_openml(name='Fashion-MNIST', version=1)
X, y = fashion.data, fashion.target.astype(int)  # X son imágenes aplanadas 28x28=784, y son etiquetas

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar datos (normalización estándar)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo
model = LogisticRegression(max_iter=1000, random_state=42)  # max_iter mayor para convergencia
model.fit(X_train_scaled, y_train)

# Predicción en datos de prueba
y_pred = model.predict(X_test_scaled)

# Métricas
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

# Prueba individual
sample = X_test_scaled[0].reshape(1, -1)
sample_pred = model.predict(sample)
print(f"Predicción para la muestra individual: {sample_pred[0]}")
print(f"Etiqueta real: {y_test[0]}")
