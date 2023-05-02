from flask import Flask, render_template, request
from tensorflow import keras
from keras.preprocessing import image
import numpy as np

# Cargar el modelo entrenado desde el archivo .h5
model = keras.models.load_model('src\modelo.h5')

# Cargar las etiquetas desde el archivo "labels.txt"
with open('src\labels.txt', 'r') as f:
    labels = f.read().splitlines()

app = Flask(__name__)

# Ruta para la página principal del sitio web
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
# código para procesar la solicitud POST
    # Obtener la imagen cargada por el usuario
    img = request.files['image']
    
    # Preprocesar la imagen para que coincida con la entrada del modelo
    img = image.load_img(img, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.0
    
    # Realizar la predicción utilizando el modelo cargado
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class = labels[predicted_class_index]
    
    # Devolver la predicción al usuario
    return predicted_class

if __name__ == '__main__':
    app.run(debug=True)
