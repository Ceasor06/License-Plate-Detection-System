import numpy as np
import matplotlib.pyplot as plt
from Levenshtein import distance as lev_distance
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

class LicensePlateEvaluator:
    def __init__(self):
        self.iou_threshold = 0.5 #Minimum IoU to set detection as correct
        self.tp = 0 #True Positives
        self.fp = 0 #False Positives
        self.fn = 0 #False Negatives
        self.tn = 0 #True Negatives

        #Initialize dictionary to store metrices for each image
        self.results = {
            'detection_iou': [],
            'ocr_accuracy': [],
            'char_accuracy': [],
            'plate_matches': [],
            'cer': [],
            'wer': [],
            'true_labels': [], 
            'predicted_scores': [], 
            'predicted_labels': [], 
        }

    def calculate_iou(self, pred_box, gt_box):
        """
        Calculate IoU between predicted and ground truth boxes
        
        """
        x1 = max(pred_box[0], gt_box[0])
        y1 = max(pred_box[1], gt_box[1])
        x2 = min(pred_box[2], gt_box[2])
        y2 = min(pred_box[3], gt_box[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        union = pred_area + gt_area - intersection
        
        return intersection / union if union > 0 else 0

    def character_error_rate(self, pred_text, gt_text):
        """Calculate Character Error Rate (CER)."""

        #Calculate the Levenshtein distance between predicted and the ground truth
        return lev_distance(pred_text, gt_text) / len(gt_text) if gt_text else 1.0

    def word_error_rate(self, pred_text, gt_text):
        """Calculate Word Error Rate (WER)."""

        #Splitting the text into words and calculating WER using Levenshtein distance
        pred_words = pred_text.split()
        gt_words = gt_text.split()
        return lev_distance(pred_words, gt_words) / len(gt_words) if gt_words else 1.0
    
    def character_accuracy(self, pred_text, gt_text):
        """Calculate character-level accuracy"""

        #Summing matching characters and accuracy based on longer length of the text
        correct_chars = sum(1 for p, g in zip(pred_text, gt_text) if p == g)
        total_chars = max(len(pred_text), len(gt_text))
        return correct_chars / total_chars if total_chars > 0 else 0

    def evaluate_single_image(self, pred_results, gt_data):
        """Evaluate detection and OCR results for a single image"""
        #Extract ground truth data
        gt_plate = gt_data['plate']
        gt_corners = np.array(gt_data['corners']).reshape(-1, 2)
        gt_box = [
            min(gt_corners[:, 0]), min(gt_corners[:, 1]),
            max(gt_corners[:, 0]), max(gt_corners[:, 1])
        ]
        
        best_match = None
        best_iou = 0
        
        #Loop through to find the one with highest IoU
        for pred in pred_results:
            pred_box = pred['coords']
            iou = self.calculate_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_match = pred
        
        #If IoU > threshold, calculate OCR accuracy and the other metrices
        if best_match and best_iou > self.iou_threshold:
            pred_text = best_match['text']
            char_acc = self.character_accuracy(pred_text, gt_plate)
            ocr_acc = 1.0 if pred_text == gt_plate else 0.0
            cer = self.character_error_rate(pred_text, gt_plate)
            wer = self.word_error_rate(pred_text, gt_plate)
            
            #Append metrics to the dictionary
            self.results['detection_iou'].append(best_iou)
            self.results['ocr_accuracy'].append(ocr_acc)
            self.results['char_accuracy'].append(char_acc)
            self.results['cer'].append(cer)
            self.results['wer'].append(wer)
            self.results['plate_matches'].append({
                'predicted': pred_text,
                'ground_truth': gt_plate,
                'iou': best_iou,
                'cer': cer,
                'wer': wer,
            })

            self.results['true_labels'].append(1) #Positive class
            self.results['predicted_scores'].append(best_iou)
            self.results['predicted_labels'].append(1 if best_iou > self.iou_threshold else 0)
        
            if pred_text == gt_plate: #Updating TP and FP counters
                self.tp += 1 #TP
            else:
                self.fp += 1 #FP
        else:
            #If no match is found, then False Negative
            self.fn += 1
            self.results['true_labels'].append(0)
            self.results['predicted_scores'].append(0.0)
            self.results['predicted_labels'].append(0)
        
        return best_match is not None

    def visualize_results(self, output_folder):
        """Create visualizations of the evaluation results"""
        os.makedirs(output_folder, exist_ok=True)
        # plt.style.use('seaborn')
        
        #1. IoU Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.results['detection_iou'], bins=20, 
                alpha=0.7, color='blue', edgecolor='black')
        plt.title('Detection IoU Distribution')
        plt.xlabel('IoU')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_folder, 'iou_distribution.png'))
        plt.close()

        #2. Confusion Matrix
        cm = confusion_matrix(self.results['true_labels'], self.results['predicted_labels'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
        plt.close()

        #3. Recognition Accuracy
        plt.figure(figsize=(10, 6))
        metrics = {
            'Full Plate': np.mean(self.results['ocr_accuracy']) * 100,
            'Character': np.mean(self.results['char_accuracy']) * 100
        }
        plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green'])
        plt.title('Recognition Accuracy')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center')
        plt.savefig(os.path.join(output_folder, 'accuracy_comparison.png'))
        plt.close()
        
        #4. Distance Distribution
        edit_distances = [lev_distance(m['predicted'], m['ground_truth']) 
                        for m in self.results['plate_matches']]
        plt.figure(figsize=(10, 6))
        plt.hist(edit_distances, bins=range(max(edit_distances) + 2), 
                alpha=0.7, color='red', edgecolor='black')
        plt.title('Edit Distance Distribution')
        plt.xlabel('Levenshtein Distance')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_folder, 'edit_distance.png'))
        plt.close()
        
        # 5. IoU vs Character Accuracy
        plt.figure(figsize=(10, 6))
        plt.scatter(self.results['detection_iou'], 
                self.results['char_accuracy'],
                alpha=0.6)
        plt.title('IoU vs Character Accuracy')
        plt.xlabel('IoU')
        plt.ylabel('Character Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, 'iou_vs_accuracy.png'))
        plt.close()
    
    def print_metrics(self):
        """Print summary metrics"""
        metrics = {
            'Mean IoU': np.mean(self.results['detection_iou']),
            'Full Plate Accuracy': np.mean(self.results['ocr_accuracy']) * 100,
            'Character Accuracy': np.mean(self.results['char_accuracy']) * 100,
            'Character Error Rate': np.mean(self.results['cer']) * 100,
            'Word Error Rate': np.mean(self.results['wer']) * 100,
            'Total Samples': len(self.results['plate_matches'])
        }

        print("\nEvaluation Metrics:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
        
        print("\nSample Predictions:")
        print("-" * 50)
        for i, match in enumerate(self.results['plate_matches'][:5]):
            print(f"Sample {i+1}:")
            print(f"Predicted: {match['predicted']}")
            print(f"Ground Truth: {match['ground_truth']}")
            print(f"IoU: {match['iou']:.2f}\n")
