import mediapipe as mp
import cv2
import os

#media pipe variables
class_hands_detect = mp.solutions.hands     # señala al modulo de herramientas para la deteccion de manos
class_drawing = mp.solutions.drawing_utils  # señala al modulo de herramientas de dibujado (puntos, lineas)

hands = class_hands_detect.Hands()          # el modelo de deteccion de manos

cap = cv2.VideoCapture(1)

cont = 0

while(True):
    ret, frame = cap.read()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copy = frame.copy()
    
    resultado = hands.process(frame_rgb)    # usando el modelo de deteccion hands busca manos en el frame
    posicion = []
    
    if resultado.multi_hand_landmarks:
        for hand in resultado.multi_hand_landmarks:
                for id_land, l_mark in enumerate(hand.landmark):    # id_land es el id del landmark de mediapipe y l_mark es la posicion del landmark
                    alto, ancho, flag = frame.shape
                    x, y = int(l_mark.x * ancho), int(l_mark.y * alto)
                    posicion.append([id_land, x, y])
                    class_drawing.draw_landmarks(frame, hand, class_hands_detect.HAND_CONNECTIONS)
                
                if len(posicion) != 0:
                    pto_i5 = posicion[9]
                    x1, y1 = (pto_i5[1] - 80), (pto_i5[2] - 80)
                    ancho_cuadro, alto_cuadro = (x1+80), (y1+80)
                    x2, y2 = (x1 + ancho_cuadro), (y1 + alto_cuadro)
                    dedos_reg = copy[y1:y2, x1:x2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    cv2.imshow("video", frame)
    k = cv2.waitKey(1)
    
    if k == 27 or cont>=300:
        break
    
cap.release()                  