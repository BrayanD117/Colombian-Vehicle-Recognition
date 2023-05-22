import os

def rename_files(directory):
    # Obtener la lista de archivos en el directorio
    file_list = os.listdir(directory)
    
    # Filtrar solo los archivos con extensión .jpg
    jpg_files = [file for file in file_list if file.endswith('.jpg')]
    
    # Ordenar los archivos alfabéticamente
    jpg_files.sort()
    
    # Renombrar los archivos
    for i, file in enumerate(jpg_files):
        # Nuevo nombre del archivo con formato "numero.jpg"
        new_name = f'descarga{i+1}.jpg'
        
        # Ruta actual y nueva ruta del archivo
        current_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)
        
        # Renombrar el archivo
        os.rename(current_path, new_path)
        print(f'Renombrado: {file} -> {new_name}')

# Directorio de los archivos .jpg
directory = 'Plates_Data\original'

# Cambiar el nombre de los archivos
rename_files(directory)
