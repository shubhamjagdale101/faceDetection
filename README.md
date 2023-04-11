# Attendance-System-Using-CNN
This is an attendance system created by image recognition

1. You can create your own dataset by executing DataCollect.py
	Just enter the name of the person and the folder will be auto created. I have coded as to capture 200photos. you can change the number of images you want. 

2. After collecting the database, run datapreprocessing.py file. It will recolor and reshape the image and will create data and target npy files for the model.

3. Then run trainingCNN.py file. It will train the CNN model. I have used 15 epoch. change it as you wish. for each epoch, a model is created. But only the best models will be saved. Among the best models, choose a suitable model by looking at accuracy ,validation loss. I have used model_11

4. To implement the model, run Recognize_Faces.py. Before that, create a Attendance.csv file. The attendence will be saved to it. 
