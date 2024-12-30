import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import cv2
from preprocess import preprocess_image

#Our custom Convolutional Neural Network for license plate OCR
class LicensePlateCNN(nn.Module):
    def __init__(self, num_classes=36, max_seq_len=10):
        super(LicensePlateCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) #1st convolutional layer
        self.pool = nn.MaxPool2d(2, 2) #Maxpooling
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) #2nd layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) #3rd layer
        self.fc1 = nn.Linear(128 * 4 * 4, 256) #Fully connected layer 1
        self.fc2 = nn.Linear(256, num_classes * max_seq_len) #Fully connected layer 2
        self.max_seq_len = max_seq_len

    # Forward Pass
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #conv1 + ReLU + pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)) #Fully connected layer 2 with ReLU
        x = self.fc2(x) #Output logits
        return x

class OCRPredictor:
    '''Custom OCRPredictor class for performing Optical Character Recognition'''
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = LicensePlateCNN() #Load the LicensePlateCNN model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)) #Load the model weights
        self.model.to(self.device)
        self.model.eval()
        
        #Image Transformations (Resizing & Normalization)
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)), #Resize image to 32*32
            transforms.ToTensor(), #Convert image to a tensor
            transforms.Normalize((0.5,), (0.5,)) #Normalize
        ])
    
    def predict(self, img):
        #Check the input image if it's anumpy array then convert it to PIL
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        img = preprocess_image(img) #Preprocess using our custom preprocessing function
        img_tensor = self.transform(img).unsqueeze(0).to(self.device) #Apply transformations to the image
        
        #Perform predictions without updating the gradients
        with torch.no_grad():
            outputs = self.model(img_tensor) #Forward pass
            outputs = outputs.view(-1, 36)
            prob = F.softmax(outputs, dim=1) #Softmax to get probabilities for each character
            pred_ind = torch.argmax(prob, dim=1).cpu().numpy()
        
        #Convert the predicted indices to characters (A-Z & 0-9)
        pred_chars = [
            chr(idx + ord('A')) if idx < 26 else chr(idx - 26 + ord('0'))
            for idx in pred_ind
        ]
        
        return ''.join(pred_chars).rstrip('A')