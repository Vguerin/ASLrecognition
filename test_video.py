#!usr/bin/python

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

Train,_,_,class_id = get_dataset(path)
inputs = keras.Inputs(Train[0].shape[:1])
#Create Model
model = ResNet50V2(
    include_top=True, weights=None, input_tensor=inputs,
    pooling=True, classes=len(class_id), classifier_activation='softmax')

model.load_weights(r'training/test_1.ckpt')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    color_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the resulting frame
    cv2.imshow('frame',color_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()