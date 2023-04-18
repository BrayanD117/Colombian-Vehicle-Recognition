from tensorflow import keras
# import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Cargar las imágenes de entrenamiento desde la carpeta "Datos" (o la ruta que corresponda)
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'Datos',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

# Cargar las etiquetas desde el archivo "labels.txt" (o la ruta que corresponda)
with open('src\labels.txt', 'r') as f:
    labels = f.read().splitlines()

# Crear el modelo
model = Sequential()
# Capas de convolución, max pooling y aplanado
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# Capas completamente conectadas
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_generator, epochs=20)

# Evaluación
# Cargar las imágenes de prueba desde la carpeta "Datos" (o la ruta que corresponda)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'Datos',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

# Obtener las etiquetas reales de prueba
y_test = test_generator.classes

# Calcular la pérdida y precisión en los datos de prueba
loss, accuracy = model.evaluate(test_generator)
print("Pérdida en los datos de prueba:", loss)
print("Precisión en los datos de prueba:", accuracy)

# Realizar predicciones en nuevos datos de entrada
# Cargar las nuevas imágenes de entrada (o la ruta que corresponda)
X_nuevos = ...  # Cargar nuevas imágenes de entrada
# Preprocesar las imágenes
X_nuevos = X_nuevos / 255
# Obtener las predicciones
predicciones = model.predict(X_nuevos)
# Obtener las clases predichas
clases_predichas = np.argmax(predicciones, axis=1)
# Imprimir las clases predichas
for i in range(len(clases_predichas)):
    print("Imagen {}: Clase predicha: {}".format(i+1, labels[clases_predichas[i]]))

########## Guardar el modelo ##########
# Guardar el modelo entrenado en un archivo
model.save('src\modelo.h5')

# Cargar un modelo guardado desde un archivo
model = keras.models.load_model('src\modelo.h5')
