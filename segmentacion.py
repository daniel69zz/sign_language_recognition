import os
import random

def limpiar_subcarpetas(carpeta_principal, num_imagenes_a_conservar=125):
    # Extensiones de imagen comunes
    extensiones_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    # Recorrer todas las subcarpetas dentro de la carpeta principal
    for raiz, _, archivos in os.walk(carpeta_principal):
        # Filtrar solo las imágenes
        imagenes = [f for f in archivos if f.lower().endswith(extensiones_validas)]
        
        # Si hay más de las necesarias, eliminar aleatoriamente
        if len(imagenes) > num_imagenes_a_conservar:
            imagenes_a_eliminar = random.sample(imagenes, len(imagenes) - num_imagenes_a_conservar)

            for imagen in imagenes_a_eliminar:
                ruta_imagen = os.path.join(raiz, imagen)
                os.remove(ruta_imagen)
                print(f"Eliminado: {ruta_imagen}")

            print(f"📂 En {raiz}, se eliminaron {len(imagenes_a_eliminar)} imágenes.")
        else:
            print(f"✅ En {raiz}, no se necesita eliminar imágenes.")

# 📌 Ejemplo de uso
carpeta_principal = (r'C:\Users\HP\Downloads\programacion_etc\sign_language\datos_valid')
limpiar_subcarpetas(carpeta_principal)
