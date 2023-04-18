import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

# Capa de convolución con 32 filtros, tamaño de filtro de 3x3 y función de activación ReLU
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
# Capa de max pooling con tamaño de pool de 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Capa de convolución con 64 filtros, tamaño de filtro de 3x3 y función de activación ReLU
model.add(Conv2D(64, (3, 3), activation='relu'))
# Capa de max pooling con tamaño de pool de 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanar la salida de la capa anterior
model.add(Flatten())

# Capa completamente conectada con 128 neuronas y función de activación ReLU
model.add(Dense(128, activation='relu'))

# Capa de salida con 3 neuronas (una para cada clase: carro, moto, camión) y función de activación softmax
model.add(Dense(3, activation='softmax'))


###### Compilacion ######
# Compilar el modelo con la función de pérdida de entropía cruzada categórica, optimizador Adam y métrica de precisión
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


###### Entrenamiento ######
# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train, epochs=10, batch_size=32)


###### Entrenamiento ######
# Evaluar el modelo con los datos de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print('Pérdida de prueba:', loss)
print('Precisión de prueba:', accuracy)

# Realizar predicciones con el modelo
predictions = model.predict(X_pred)



# Guardar el modelo entrenado en un archivo
model.save('modelo.h5')

# Cargar un modelo guardado desde un archivo
model = keras.models.load_model('modelo.h5')
