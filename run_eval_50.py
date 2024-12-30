import os
from pipeline import process #Importing the process function from pipeline.py
from evaluation import LicensePlateEvaluator #Importing the Licenseplate evaluater class from evaluation.py
import cv2
import json
import csv
import matplotlib.pyplot as plt
from run_eval import load_ground_truth
import numpy as np
from tqdm import tqdm
# import pytesseract

def evaluate_dataset(base_dir, yolo_model_path, ocr_model_path, evaluated_images_dir):
    """
    This function processes the dataset of licenseplates images to evaluate 
    the performance of our license plate detection + our custom OCR (given in pipeline.py)
    
    """
    evaluator = LicensePlateEvaluator() #Initialize the evaluator
    os.makedirs(evaluated_images_dir, exist_ok=True)

    #Define folders to process
    all_files = os.listdir(base_dir)
    img_files = sorted([f for f in all_files if f.endswith('.png')])
    total_processed = 0 #Counter to keep track of processed images
    
    for img_file in tqdm(img_files):  #Adjusted for preprocessing 50 images
        img_path = os.path.join(base_dir, img_file)
        txt_path = img_path.replace('.png', '.txt') #Ground truth for images we are using
        
        if os.path.exists(img_path) and os.path.exists(txt_path):
            try: 
                gt_data = load_ground_truth(txt_path) #Load the ground truth data from the txt file in the dataset

                #Process function to get the detected plates
                results, image = process(
                    img_path,
                    yolo_model_path,
                    ocr_model_path
                )
                #Evaluate the results for the image against the ground truth
                evaluator.evaluate_single_image(results, gt_data)

                evaluated_image_path = os.path.join(evaluated_images_dir, img_file)
                cv2.imwrite(evaluated_image_path, image)

                total_processed += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print(f"\nTotal images processed: {total_processed}")
    return evaluator

#Function to save the evaluation results to csv file
def save_results_to_csv(results, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['IoU', 'Predicted', 'Ground_Truth', 'Char_Accuracy', 'CER', 'WER', 'Full_Plate_Match'])
        
        for i in range(len(results['detection_iou'])):
            writer.writerow([
                results['detection_iou'][i],
                results['plate_matches'][i]['predicted'],
                results['plate_matches'][i]['ground_truth'],
                results['char_accuracy'][i],
                results['cer'][i],
                results['wer'][i],
            ])

#Function to save the results to a JSON file
def save_results_to_json(results, output_path):
    results_dict = {
        'metrics': {
            'mean_iou': float(np.mean(results['detection_iou'])), #Mean Intersection over Union (IoU)
            'full_plate_accuracy': float(np.mean(results['ocr_accuracy']) * 100),
            'character_accuracy': float(np.mean(results['char_accuracy']) * 100),
            'mean_cer': float(np.mean(results['cer']) * 100), #Mean Character Error Rate (CER)
            'mean_wer': float(np.mean(results['wer']) * 100), #Mean Word Error Rate (WER)
        },
        #Detailed result for each image
        'detailed_results': {
            'detection_iou': results['detection_iou'],
            'ocr_accuracy': results['ocr_accuracy'],
            'char_accuracy': results['char_accuracy'],
            'cer': results['cer'],
            'wer': results['wer'],
            'plate_matches': results['plate_matches']
        }
    }
    with open(output_path, 'w') as jsonfile:
        json.dump(results_dict, jsonfile, indent=4)

if __name__ == "__main__":
    base_dir = "./dataset"
    yolo_model_path = "./exp4/weights/best.pt"
    ocr_model_path = "./BEST_cnn.pth"
    results_dir = "./results50"
    evaluated_images_dir = "./evaluated_images"
    
    os.makedirs(results_dir, exist_ok=True)
    
    #Run the evaluation
    evaluator = evaluate_dataset(base_dir, yolo_model_path, ocr_model_path, evaluated_images_dir)
   
    #Save results
    save_results_to_csv(evaluator.results, os.path.join(results_dir, 'evaluation_results.csv'))
    save_results_to_json(evaluator.results, os.path.join(results_dir, 'evaluation_results.json'))

    # visualize_results(evaluator, os.path.join(results_dir, 'graphs'))
    graphs_dir = os.path.join(results_dir, 'graphs')
    evaluator.visualize_results(graphs_dir)
    
    print("\nCustom OCR Metrics:")
    print("-" * 50)
    print(f"Mean IoU: {np.mean(evaluator.results['detection_iou']):.2f}")
    print(f"Full Plate Accuracy: {np.mean(evaluator.results['ocr_accuracy']) * 100:.2f}%")
    print(f"Character Accuracy: {np.mean(evaluator.results['char_accuracy']) * 100:.2f}%")