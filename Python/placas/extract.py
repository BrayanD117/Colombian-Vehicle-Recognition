import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def extract_characters(image_path, min_area, max_area, min_aspect_ratio, max_aspect_ratio):
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

        if min_area < cv2.contourArea(contour) < max_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            # Ajustar el tamaño del contorno para obtener solo el carácter
            roi = image[y:y + h, x:x + w]
            resized_character = cv2.resize(roi, (100, 100))  # Ajustar el tamaño a tus necesidades

            # Agregar el carácter a la lista de imágenes
            character_images.append(resized_character)

    return character_images

def process_image():
    image_path = 'Plates_Data/original/descarga1.jpg'
    characters = extract_characters(image_path, min_area_slider.get(), max_area_slider.get(), min_aspect_ratio_slider.get(), max_aspect_ratio_slider.get())
    
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


def show_image(image_path):
    image = Image.open(image_path)
    image.thumbnail((300, 300))  # Ajustar el tamaño de la imagen
    photo = ImageTk.PhotoImage(image)
    image_label.configure(image=photo)
    image_label.image = photo

# Crear la ventana principal
window = tk.Tk()
window.title('Ajuste de Parámetros')
window.geometry('600x400')

# Crear los controles deslizantes
min_area_slider = tk.Scale(window, from_=0, to=2000, length=300, orient='horizontal')
max_area_slider = tk.Scale(window, from_=0, to=4000, length=300, orient='horizontal')
min_aspect_ratio_slider = tk.Scale(window, from_=0, to=100, resolution=1, length=300, orient='horizontal')
max_aspect_ratio_slider = tk.Scale(window, from_=0, to=5000, resolution=1, length=300, orient='horizontal')

# Agregar etiquetas a los controles desliz
tk.Label(window, text='Min Area').pack()
min_area_slider.pack()
tk.Label(window, text='Max Area').pack()
max_area_slider.pack()
tk.Label(window, text='Min Aspect Ratio').pack()
min_aspect_ratio_slider.pack()
tk.Label(window, text='Max Aspect Ratio').pack()
max_aspect_ratio_slider.pack()

# Agregar botón para procesar la imagen
ttk.Button(window, text='Procesar Imagen', command=process_image).pack()

# Etiqueta para mostrar la imagen
image_label = tk.Label(window)
image_label.pack()

# Mostrar la imagen inicial
initial_image_path = 'Plates_Data/original/descarga1.jpg'
show_image(initial_image_path)

# Ejecutar la ventana principal
window.mainloop()
