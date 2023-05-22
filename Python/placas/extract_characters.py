import cv2
import numpy as np

def extract_characters(image_path):
    # Cargar la imagen en escala de grises
    image = cv2.imread(image_path, 0)

    # Aplicar un umbral adaptativo para convertir la imagen en blanco y negro
    _, threshold = cv2.threshold(image, 0, 128, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar los contornos de los caracteres
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar y extraer los caracteres individuales
    character_images = []
    for contour in contours:
        # Obtener las coordenadas del rectángulo que encierra el contorno
        x, y, w, h = cv2.boundingRect(contour)

        # Filtrar los caracteres basado en el tamaño y la relación de aspecto
        aspect_ratio = w / float(h)
        min_area = 5
        max_area = 5000
        min_aspect_ratio = 0.1
        max_aspect_ratio = 100.5


        if min_area < cv2.contourArea(contour) < max_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            # Ajustar el tamaño del contorno para obtener solo el carácter
            roi = image[y:y + h, x:x + w]
            resized_character = cv2.resize(roi, (100, 100))  # Ajustar el tamaño a tus necesidades

            # Agregar el carácter a la lista de imágenes
            character_images.append(resized_character)

    return character_images

# Ruta de la imagen de entrada
image_path = 'Plates_Data\original\descarga18.jpg'
print(image_path)  # Verificar la ruta de la imagen de entrada
image = cv2.imread(image_path, 0)


# Extraer los caracteres de la imagen
characters = extract_characters(image_path)

# Verificar si se encontraron caracteres
if len(characters) == 0:
    print("No se encontraron caracteres que cumplan los criterios de filtrado.")
else:
    # Guardar los caracteres en imágenes individuales
    for i, character in enumerate(characters):
        # Ruta de salida de la imagen del carácter
        output_path = f'characters/caracter_{i}.jpg'

        # Guardar la imagen del carácter
        print(output_path)  # Verificar la ruta de salida
        cv2.imwrite(output_path, character)
