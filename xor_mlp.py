import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Dataset XOR
x_train = np.array([[0,0],[0,1],[1,0],[1,1]], dtype="float32")
y_train = np.array([[0],[1],[1],[0]], dtype="float32")

# Modelo MLP
model = keras.Sequential([
    layers.Dense(2, input_dim=2, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compilar modelo
model.compile(optimizer=keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Entrenamiento
fit_history = model.fit(x_train, y_train, epochs=50, batch_size=4)

# Graficar curva de pérdida
plt.plot(fit_history.history['loss'], label='Pérdida')
plt.legend()
plt.title('Resultado del Entrenamiento')
plt.show()

# Pesos y bias
weights_HL, biases_HL = model.layers[0].get_weights()
weights_OL, biases_OL = model.layers[1].get_weights()
print(weights_HL, biases_HL, weights_OL, biases_OL)

# Predicciones
prediccion = model.predict(x_train)
print(prediccion)
