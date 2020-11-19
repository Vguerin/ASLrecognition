#/usr/bin/env python

import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),-1)
        img = cv2.resize(img, (299,299), interpolation=cv2.INTER_CUBIC)
        img_color = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if img is not None:
            images.append(img_color)
    return images

if __name__ == "__main__": 
    img = load_images_from_folder("../handson-ml/animaux")