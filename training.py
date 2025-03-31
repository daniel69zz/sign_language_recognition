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

datos_train = (r'C:\Users\HP\Downloads\ASL_Alphabet_Dataset\asl_alphabet_train')
datos_valid = (r'C:\Users\HP\Downloads\ASL_Alphabet_Dataset\asl_alphabet_test')

#   PARAMETROS
epochs_ = 20
altura, longitud = 200, 200
batch_size = 1
pasos = int(300/1)
pasos_valid = int(300/1)
filtros_conv1 = 32
filtros_conv2 = 64

size_filtro1 = (3, 3)
size_filtro2 = (2, 2)
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


#   APLANANDO A 1D LA IMAGEN (MATRIZ)
cnn.add(Flatten())
cnn.add(Dense(1383, activation = 'relu'))
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