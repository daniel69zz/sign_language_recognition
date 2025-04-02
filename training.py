import cv2
import os

#       PARA EL ENTRENAMIENTO DEL MODELO
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as k
from tensorflow.keras.optimizers import Adam

datos_train = (r'C:\Users\HP\Downloads\programacion_etc\sign_language\datos_train')
datos_valid = (r'C:\Users\HP\Downloads\programacion_etc\sign_language\datos_valid')

#   PARAMETROS
epochs_ = 110   #para mas de 200k imagenes, el dataset tiene 213k imagenes aproximadamente
altura, longitud = 200, 200
batch_size = 64
pasos = 2656
pasos_valid = 672
filtros_conv1 = 32
filtros_conv2 = 64
filtros_conv3 = 128
filtros_conv4 = 256



size_filtro1 = (3, 3)
size_filtro2 = (3, 3)
size_pooling = (2, 2)

clases = 27
lr = 0.0005

# Pre processing de imagenes
pre_process_train = ImageDataGenerator(rescale = 1./255, shear_range = 0.3, zoom_range = 0.3, horizontal_flip = True)
pre_process_valid = ImageDataGenerator(rescale = 1./255)

# RELACIONAR LAS IMAGENES CON SU RESPECTIVO TARGET DE LA CARPETA
image_train = pre_process_train.flow_from_directory(datos_train, target_size = (altura, longitud), batch_size = batch_size, class_mode = 'categorical')

image_valid = pre_process_valid.flow_from_directory(datos_valid, target_size = (altura, longitud), batch_size = batch_size, class_mode = 'categorical')

#               CREACION DEL MODELO CNN
cnn = Sequential()

cnn.add(Conv2D(filtros_conv1, size_filtro1, padding = 'same', input_shape = (altura, longitud, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = size_pooling))

cnn.add(Conv2D(filtros_conv2, size_filtro2, padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = size_pooling))

cnn.add(Conv2D(filtros_conv3, size_filtro2, padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = size_pooling))

cnn.add(Conv2D(filtros_conv4, size_filtro2, padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = size_pooling))


#   APLANANDO A 1D LA IMAGEN (MATRIZ)
cnn.add(Flatten())
cnn.add(Dense(130, activation = 'relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))


# OPTIMIZADOR
optimizar = Adam(learning_rate= lr)
cnn.compile(loss = 'categorical_crossentropy', optimizer= optimizar, metrics=['accuracy'])

#Entrenaremos nuestra red
cnn.fit(image_train, steps_per_epoch=pasos, epochs= epochs_, validation_data= image_valid, validation_steps=pasos_valid)

#Guardamos el modelo
cnn.save('Modelo.h5')
cnn.save_weights('pesos.h5')