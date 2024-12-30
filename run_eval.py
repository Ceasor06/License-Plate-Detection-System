from pipeline import process
from evaluation import LicensePlateEvaluator
import matplotlib.pyplot as plt
import cv2

def load_ground_truth(annotation_path):
    """Load ground truth data from annotation file"""
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    gt_data = {}
    plate = None
    corners = None   
    
    for line in lines:
        if line.startswith('plate:'):
            plate = line.split(':')[1].strip()
        elif line.startswith('corners:'):
            #Split by spaces first to separate coordinate pairs
            corner_pairs = line.split(':')[1].strip().split()
            #Convert each "x,y" string into [x,y] coordinates
            corners = []
            for pair in corner_pairs:
                x, y = map(float, pair.split(','))
                corners.extend([x, y])
    
    return {
        'plate': plate,
        'corners': corners
    }

def run_evaluation(image_path, annotation_path, yolo_model_path, ocr_model_path):
    #Initialize evaluator
    evaluator = LicensePlateEvaluator()
    
    #Load ground truth
    gt_data = load_ground_truth(annotation_path)
    
    #Run detection and OCR
    results, image = process(
        image_path, 
        yolo_model_path, 
        ocr_model_path,
        output_path="./evaluated_output.jpg"
    )
    
    #Evaluate results
    evaluator.evaluate_single_image(results, gt_data)
    
    #Display results
    evaluator.print_metrics()
    evaluator.visualize_results()
    
    #Display the image with annotations
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Detection and OCR Results")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image_path = "./dataset/track0002[02].png"
    annotation_path = "./dataset/track0002[02].txt"
    yolo_model_path = "./exp4/weights/best.pt"
    ocr_model_path = "./BEST_cnn.pth"
    
    run_evaluation(image_path, annotation_path, yolo_model_path, ocr_model_path)