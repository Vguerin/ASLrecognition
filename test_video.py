#!usr/bin/python

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from generate_dataset import get_dataset

def prepare_image(image, target_width = 224, target_height = 224):
    
    image_resized_grey = cv2.resize(image,(target_width,target_height))
    image_resized = cv2.cvtColor(image_resized_grey,cv2.COLOR_BGR2RGB)

    return image_resized

Train,_,_,class_id = get_dataset('dataset')
inputs = keras.Input(Train[0].shape[1:])
#Create Model
model = ResNet50V2(
    include_top=True, weights=None, input_tensor=inputs,
    pooling=True, classes=len(class_id), classifier_activation='softmax')

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.load_weights(r'training/test_1.ckpt')

keys_list = list(class_id.keys())

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    width = frame.shape[0]
    height = frame.shape[1]

    # if not frame:
    #     print("No image")
    #     break
    # Our operations on the frame come here
    prepare_img = prepare_image(frame)
    prepare_img = tf.expand_dims(prepare_img, axis=0)
    proba = model.predict(prepare_img)

#     if not proba:
#         continue
    print(proba)
    pred = np.argmax(proba)
    text = "prediction is " + keys_list[pred]
    print(text)
    #print("pred : ", keys_list[pred])

    # Display the resulting frame
    cv2.imshow('frame',frame)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(frame,text,(width/2,height-20), font, .5,(255,255,255),2,cv2.LINE_AA)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()