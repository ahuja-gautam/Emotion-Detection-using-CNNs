# Emotion Detection with CNNs

Use your webcam to predict emotions in real time!

Dataset Used: [FER dataset (Kaggle)](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
Please download the dataset and extract it in the master directory if you wish to train the model yourself. (It was trained on Google Colab so please make necessary changes in the directory structure if you wish to train it locally)

# Usage
1)Download 'model_1.h5' and main.ipynb  The model is stored in the .h5 file

2)Run main.ipynb and place your face in the bounding box

3)Please ensure your surroundings are well lit and your face is evenly lit for best results.

4)The video recording of your emotions is saved in 'Emotions.mp4v'. If you do not want this feature comment out the lines:

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('Emotions.mp4v', fourcc, 20.0, (640, 480))
out.release()

# Features
-Accuracy of 63.2% achieved which would place us [9th in the Leaderboard](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/leaderboard)

-Dataset split into 90% train: 10% test

-Used Data Augmentation including but not limited to rotation and zooming of image

-Normalised dataset between [0,1] to reduce computation load.

# Model Architecture
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 46, 46, 64)        640       
_________________________________________________________________
batch_normalization_1 (Batch (None, 46, 46, 64)        256       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 23, 23, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 23, 23, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 21, 21, 128)       73856     
_________________________________________________________________
batch_normalization_2 (Batch (None, 21, 21, 128)       512       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 10, 10, 128)       0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 10, 10, 128)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 512)         590336    
_________________________________________________________________
batch_normalization_3 (Batch (None, 8, 8, 512)         2048      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 512)         0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 4, 4, 512)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 2, 2, 512)         2359808   
_________________________________________________________________
batch_normalization_4 (Batch (None, 2, 2, 512)         2048      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 1, 1, 512)         0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 1, 1, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 260)               133380    
_________________________________________________________________
dropout_5 (Dropout)          (None, 260)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 510)               133110    
_________________________________________________________________
dropout_6 (Dropout)          (None, 510)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 7)                 3577      
=================================================================
Total params: 3,299,571
Trainable params: 3,297,139
Non-trainable params: 2,432
```
# Findings
![Confusion Matrix:](https://github.com/ahuja-gautam/Emotion-Detection-via-Webcam/blob/master/ConfusionMatrix.PNG)


-Disgust has very few samples in the dataset. There is also some misclassification between Disgust and Anger which show similar features like raised eyebrows.

-Anger in some cases is classified as Neutral

-Fear is classified mostly between Fear, Sad, and Neutral

-Sad is also sometimes classified as Neutral

These issues mostly arise due to the quality of the dataset and due to the limitations of CNN



