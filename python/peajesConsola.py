from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Desactivar la notacion cientifica para mayor claridad en la consola :)
np.set_printoptions(suppress=True)

# Cargar el modelo
model = load_model("src\keras_model.h5", compile=False)

# Cargar las etiquetas de reconocimiento
class_names = open("src\labels.txt", "r").readlines()

# Crear la matriz de la forma correcta para alimentar el modelo de keras
# La 'longitud' o el número de imágenes que puede colocar en la matriz es
# determinado por la primera posición en la tupla de forma, en este caso 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Ruta de la imagen a verificar que se solicita al usuario
image_path = input("Ingrese la ruta de la imagen: ")

# Abre el archivo de imagen
with Image.open(image_path) as image:
    # cambiar el tamaño de la imagen para que tenga al menos 224x224 y luego recortar desde el centro
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)

    # convertir la imagen en una matriz numpy
    image_array = np.asarray(image)

    # Normalizar la imagen
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Carga la imagen en la matriz
    data[0] = normalized_image_array

    # Predecir el tipo de vehiculo usando el modelo entrenado
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Imprime el tipo de vehiculo y el puntaje de confianza
    print(f"Class: {class_name}, Confidence Score: {confidence_score}")