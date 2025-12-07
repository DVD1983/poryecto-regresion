#modificacionde prueba

 modelo = models.regretion

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Datos de ejemplo
X = np.array([[1], [2], [3], [4], [5]])   # variable independiente
y = np.array([2, 4, 5, 4, 5])             # variable dependiente

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Mostrar coeficientes
print("Pendiente:", modelo.coef_[0])
print("Intersección:", modelo.intercept_)

# Predecir
X_nuevo = np.array([[6]])
prediccion = modelo.predict(X_nuevo)
print("Predicción para X=6:", prediccion[0])

# Graficar
plt.scatter(X, y)
plt.plot(X, modelo.predict(X))
plt.xlabel("X")
plt.ylabel("y")
plt.title("Regresión Lineal
