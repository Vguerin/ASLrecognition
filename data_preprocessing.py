import cv2

def prepare_image(image, target_width = 120, target_height = 120):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image * (1.0/255.0)
    image_resized = cv2.resize(image,(target_width,target_height))
    return image_resized