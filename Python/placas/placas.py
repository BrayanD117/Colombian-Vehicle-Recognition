import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical

# Preprocesamiento de datos
def preprocess_data(image_paths, labels, target_size, grayscale=False):
    num_samples = len(image_paths)
    if grayscale:
        channels = 1
    else:
        channels = 3

    X = np.zeros((num_samples, target_size[0], target_size[1], channels))
    y = np.zeros((num_samples, len(labels)))

    for i, path in enumerate(image_paths):
        img = load_img(path, target_size=target_size, grayscale=grayscale)
        img = img_to_array(img)
        img = img.astype('float32') / 255.0  # Normalización de los valores de píxel

        X[i] = img
        y[i, labels.index(labels[i])] = 1

    return X, y

# Dimensiones de entrada del modelo
input_shape = (32, 32)  # Por ejemplo, 32x32 píxeles

# Datos a color
x_train_color, y_train = preprocess_data(train_image_paths, train_labels, input_shape)
x_test_color, y_test = preprocess_data(test_image_paths, test_labels, input_shape)

# Datos en blanco y negro
x_train_bw, y_train = preprocess_data(train_image_paths, train_labels, input_shape, grayscale=True)
x_test_bw, y_test = preprocess_data(test_image_paths, test_labels, input_shape, grayscale=True)

# Creación del modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape[0], input_shape[1], 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(x_train_color, y_train, batch_size=32, epochs=10, validation_data=(x_test_color, y_test))

# Evaluación del modelo en el conjunto de prueba
score = model.evaluate(x_test_color, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
