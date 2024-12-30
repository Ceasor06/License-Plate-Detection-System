import torch
import cv2

class LicensePlateDetector:
    def __init__(self, model_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    
    def detect(self, image_path):
        image = cv2.imread(image_path)
        results = self.model(image) #YOLO model to detect objects
        detections = results.pandas().xyxy[0] #Detected objects as a pandas dataframe
        plates = []
        
        for idx, row in detections.iterrows():
            if int(row['class']) == 1: #Checking if the detected object is a license plate i.e. class 1 

                #Coordinates of the bounding box
                x_min, y_min, x_max, y_max = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])

                #Extract the license plate region using above bounding box coordinate
                plate = image[y_min:y_max, x_min:x_max]

                #Append the detected plate and it's coordinates
                plates.append({
                    'plate': plate,
                    'coords': (x_min, y_min, x_max, y_max)
                })
        
        return plates, image