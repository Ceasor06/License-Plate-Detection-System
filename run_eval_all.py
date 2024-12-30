import os
from pipeline import process
from evaluation import LicensePlateEvaluator
import cv2
import json
import csv
import matplotlib.pyplot as plt
from run_eval import load_ground_truth
import numpy as np
from tqdm import tqdm
import easyocr

def evaluate_dataset(base_dir, yolo_model_path, ocr_model_path):
    #Initialize evaluators for our custom OCR model and EasyOCR
    evaluator = LicensePlateEvaluator()
    easyocr_evaluator = LicensePlateEvaluator()

    reader = easyocr.Reader(['en'], gpu=True) #EasyOCR reader

    folders = {
        'training': range(1, 61),
        'validation': range(61, 91),
        'testing': range(91, 151)
    }
    
    total_processed = 0 #Counter for total processed image
    
    #Iterate over trainig, testing and validation folders
    for folder_type, track_range in folders.items():
        print(f"\nProcessing {folder_type} split...")
        folder_path = os.path.join(base_dir, folder_type)
        
        for track_num in tqdm(track_range):
            track_folder = f'track{track_num:04d}'
            track_path = os.path.join(folder_path, track_folder)
            
            #Process each image in track (1-60)
            for img_idx in range(1, 61):
                img_name = f'{track_folder}[{img_idx:02d}].png'
                img_path = os.path.join(track_path, img_name)
                txt_path = img_path.replace('.png', '.txt')
                
                if os.path.exists(img_path) and os.path.exists(txt_path):
                    try:
                        gt_data = load_ground_truth(txt_path) #ground truth data for the current image

                        #Process images using the custom OCR model
                        results, image = process(
                            img_path,
                            yolo_model_path,
                            ocr_model_path
                        )
                        evaluator.evaluate_single_image(results, gt_data)

                        #Process image using EasyOCR 
                        easyocr_results = process_with_easyocr(img_path, reader)
                        easyocr_evaluator.evaluate_single_image(easyocr_results, gt_data)
                        total_processed += 1
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
    
    print(f"\nTotal images processed: {total_processed}")
    return evaluator, easyocr_evaluator

def process_with_easyocr(img_path, reader):
    """Process image with EasyOCR"""
    image = cv2.imread(img_path)
    results = reader.readtext(image)
    
    processed_results = []

    #Process EasyOCR results:
    for (bbox, text, prob) in results:
        #Convert EasyOCR bbox format to [x1,y1,x2,y2]
        coords = [
            min(point[0] for point in bbox),
            min(point[1] for point in bbox),
            max(point[0] for point in bbox),
            max(point[1] for point in bbox)
        ]
        processed_results.append({
            'coords': coords,
            'text': text,
            'confidence': prob
        })
    
    return processed_results

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
                results['wer'][i]
            ])

def save_results_to_json(results, output_path):
    results_dict = {
        'metrics': {
            'mean_iou': float(np.mean(results['detection_iou'])),
            'full_plate_accuracy': float(np.mean(results['ocr_accuracy']) * 100),
            'character_accuracy': float(np.mean(results['char_accuracy']) * 100),
            'mean_cer': float(np.mean(results['cer']) * 100),
            'mean_wer': float(np.mean(results['wer']) * 100),
    },
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

def visualize_results(evaluator, easyocr_evaluator, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    #IoU Distribution comparison
    plt.hist(evaluator.results['detection_iou'], bins=20, 
             alpha=0.5, label='Custom OCR', color='blue')
    plt.hist(easyocr_evaluator.results['detection_iou'], bins=20, 
             alpha=0.5, label='EasyOCR', color='red')
    plt.title('Detection IoU Distribution Comparison')
    plt.xlabel('IoU')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'iou_comparison.png'))
    plt.close()
    
    #Recognition Accuracy Comparison
    plt.figure(figsize=(10, 6))
    metrics = {
        'Full Plate - Custom': np.mean(evaluator.results['ocr_accuracy']) * 100,
        'Character - Custom': np.mean(evaluator.results['char_accuracy']) * 100,
        'Full Plate - EasyOCR': np.mean(easyocr_evaluator.results['ocr_accuracy']) * 100,
        'Character - EasyOCR': np.mean(easyocr_evaluator.results['char_accuracy']) * 100
    }
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'lightblue', 'red', 'pink'])
    plt.title('Recognition Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'accuracy_comparison.png'))
    plt.close()

if __name__ == "__main__":
    base_dir = "./UFPR-ALPR dataset"
    yolo_model_path = "./exp4/weights/best.pt"
    ocr_model_path = "./BEST_cnn.pth"
    results_dir = "./resultsAll"
    
    os.makedirs(results_dir, exist_ok=True)
    
    #Run evaluation
    evaluator, easyocr_evaluator = evaluate_dataset(base_dir, yolo_model_path, ocr_model_path)
    
    #Save results
    save_results_to_csv(evaluator.results, os.path.join(results_dir, 'evaluation_results.csv'))
    save_results_to_json(evaluator.results, os.path.join(results_dir, 'evaluation_results.json'))
    save_results_to_csv(easyocr_evaluator.results, os.path.join(results_dir, 'easyocr_evaluation_results.csv'))
    save_results_to_json(easyocr_evaluator.results, os.path.join(results_dir, 'easyocr_results.json'))

    #Generate visualizations
    visualize_results(evaluator, easyocr_evaluator, os.path.join(results_dir, 'graphs'))
    graphs_dir_custom = os.path.join(results_dir, 'graphs_custom')
    graphs_dir_easyocr = os.path.join(results_dir, 'graphs_easyocr')
    evaluator.visualize_results(graphs_dir_custom)
    easyocr_evaluator.visualize_results(graphs_dir_easyocr)
    
    #Print final metrics for both custom OCR and EasyOCR
    print("\nCustom OCR Metrics:")
    print("-" * 50)
    print(f"Mean IoU: {np.mean(evaluator.results['detection_iou']):.2f}")
    print(f"Full Plate Accuracy: {np.mean(evaluator.results['ocr_accuracy']) * 100:.2f}%")
    print(f"Character Accuracy: {np.mean(evaluator.results['char_accuracy']) * 100:.2f}%")
    
    print("\nEasyOCR Metrics:")
    print("-" * 50)
    print(f"Mean IoU: {np.mean(easyocr_evaluator.results['detection_iou']):.2f}")
    print(f"Full Plate Accuracy: {np.mean(easyocr_evaluator.results['ocr_accuracy']) * 100:.2f}%")
    print(f"Character Accuracy: {np.mean(easyocr_evaluator.results['char_accuracy']) * 100:.2f}%")