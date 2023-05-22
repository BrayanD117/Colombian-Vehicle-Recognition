# Codigo para cambiar el color de las imagenes a blanco y negro de las placas y asi lograr una extension de datos

from PIL import Image
import os

def convert_to_bw(input_dir, output_dir):
    if not os.path.exists(output_dir):  # Verificar si el directorio de salida existe
        os.makedirs(output_dir)  # Crear el directorio de salida si no existe

    for filename in os.listdir(input_dir):  # Iterar sobre los archivos del directorio de entrada
        input_path = os.path.join(input_dir, filename)  # Ruta completa del archivo de entrada
        output_path = os.path.join(output_dir, filename)  # Ruta completa del archivo de salida

        try:
            with Image.open(input_path) as image:  # Abrir la imagen utilizando Pillow
                bw_image = image.convert('L')  # Convertir la imagen a blanco y negro
                bw_image.save(output_path)  # Guardar la imagen convertida en el directorio de salida
                print(f"Imagen {filename} convertida a blanco y negro.")
        except Exception as e:
            print(f"Error al procesar la imagen {filename}: {str(e)}")

# Directorio de entrada con las imágenes a convertir
input_directory = 'Plates_Data\original'

# Directorio de salida para las imágenes en blanco y negro
output_directory = 'Plates_Data\B-W'

# Llamada a la función para convertir las imágenes
convert_to_bw(input_directory, output_directory)
