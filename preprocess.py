import os
import cv2
import numpy as np
from PIL import Image, ImageFilter

def preprocess_image(image, output_size=(32, 32), save_intermediate=False, save_dir="preprocessed_images"):
    if save_intermediate and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if isinstance(image, str):
        image = cv2.imread(image)
    
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:   #Only height and width
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) #Convert to RGB
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert BGR to RGB

        image = Image.fromarray(image) #Convert NumPy array to PIL Image
 
    #Convert image to grayscale for further processing
    grayscale_image = image.convert('L') #L mode for grayscaling
    if save_intermediate:
        grayscale_image.save(os.path.join(save_dir, "grayscale_image.png"))

    #Apply sharpening filter to enhance edges and text clarification
    sharpened = grayscale_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    if save_intermediate:
        sharpened.save(os.path.join(save_dir, "sharpened_image.png"))
    
    #Resize the sharpened image to the target output size
    resized = sharpened.resize(output_size)
    if save_intermediate:
        resized.save(os.path.join(save_dir, "resized_image.png"))
    
    return resized

