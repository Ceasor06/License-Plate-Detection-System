FILE DESCRIPTION:
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
We have created a code pipeline where we have divided our functionalities into seperate files.
Preprocess.py deals with the pre processing of the license plate.
Detector.py uses the Yolov5 model's saved weights to extract the vehicle and the license plate.
ocr_model.py file contains our custom ocr model.
pipeline.py file combines these three files and draws the recognized text on a new image.

DATASET:
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
As the dataset was over 10 GB in size, we have not added it in the code. 
We have attached a link to the database in the report. https://github.com/raysonlaroca/ufpr-alpr-dataset/blob/master/license-agreement.md

We have picked 50 images randomly from the original DB and added that to the dataset folder. The folder has an image pair and its text file which contains
the ground truth of the vehicle and license plate.

HOW TO RUN:
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
One can run simply by running the file run_eval_50.py. The required libraries are mentioned in the requirements.txt 
which would be needed to run the code.

To run the code via bash type: "bash run.sh" in the terminal when present inside the project directory. The code will run in approximately 30 seconds. The results will
be stored in the results50 folder. There is a folder for graphs, csv and json with results for each image. The evaluated_images folder stores all the pictures
with the predicted license plate and characters.

If you want to run the algorithm for a single image, you can first pipeline.py. To get graphs and detailed results, run run_eval.py. 
Right now it is taking track0002[02].png as the input. 

RESULTS FOR ENTIRE DB
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
We have also added the code with which we ran our algorithm for the entire dataset. We have also compared our custom ocr with easyocr in the file
run_eval_all. As we have not attached the entire database, it wont run. Once the dataset is downloaded, the script will run. It takes around 
2 hours as we are going through 4500 images two times, using our ocr model followed by easyOCR. Its output is in the folder resultsEntireDB.