import cv2
from preprocess import preprocess_image
from detector import LicensePlateDetector
from ocr_model import OCRPredictor

def process(image_path, yolo_model_path, ocr_model_path, output_path=None):
    #Initialize our finetuned YOLOv5 and the ocr model
    detector = LicensePlateDetector(yolo_model_path)
    ocr = OCRPredictor(ocr_model_path)
    
    #Detect license plates
    plates, original_image = detector.detect(image_path)
    
    #Process each detected plate
    results = []
    for plate_info in plates:
        plate_img = plate_info['plate']
        coords = plate_info['coords'] #Bounding box coordinates
        
        # plate_img = preprocess_image(plate_img, save_intermediate=True)
        plate_img = preprocess_image(plate_img) #Using custom preprocess tp preprocess the license plate

        #Using custom OCR model to recognize the text and append to the results
        text = ocr.predict(plate_img)
        results.append({'text': text, 'coords': coords})
        
        #Draw a rectangle box around the detected license plate on the original image
        x_min, y_min, x_max, y_max = coords
        cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        #Add the recognized text on top pf the bounding box
        cv2.putText(original_image, text, (x_min, y_min-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, original_image)
    
    return results, original_image

if __name__ == "__main__":
    image_path = "./dataset/track0002[02].png"
    yolo_model_path = "./exp4/weights/best.pt"
    ocr_model_path = "./BEST_cnn.pth"
    output_path = "./output_single.jpg"
    
    results, image = process(image_path, yolo_model_path, ocr_model_path, output_path)
    print("Detected license plates:", [r['text'] for r in results])