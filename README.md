# Gender, Age and Emotion Analysis By Image with Deep Learnig
This project was developed by me and my friend [Fadime Coşkuner](https://github.com/FadimeCoskuner). This project we have developed is our final project.
*You can review our thesis from the file named MeryemSahin_FadimeCoskuner.pdf.*

## Description:
During the analysis process, the detection of faces in the images and the determination of characteristics such as age, gender, and emotion of these faces are targeted. These analyses are conducted by examining facial features and expressions. For example, age analysis takes into account wrinkles and signs of aging on the face. Gender analysis, on the other hand, considers facial features and expressions. Emotion analysis is performed by observing changes in facial expressions.

## Installations:
Install dependencies using requirements.txt
```
pip install -r requirements.txt
```

## Usage:
By starting the webcam, the program first detects and frames the person's face, then makes predictions about the person's gender, age, and mood. For this, first of all:
- The following file should be compiled for sentiment analysis.
```
emotion_detection.py
```
- Likewise, compilation should be made in terms of gender and age.
```
age_gender_detection.py
```
- As a result of these compilations, three .h5 files should be obtained. For example 100 epochs:
```
age_model_100epochs.h5
gender_model_100epochs.h5
emotion_detection_model_100epochs.h5
```
- These .h5 files obtained should be given to the program.
```
live_age_gender_emotion_detection.py
```
- That's it!

## Datasets:
- We used the fer2013 dataset for sentiment analysis. You can download it [here](https://www.kaggle.com/datasets/msambare/fer2013)!
- We worked with UTKFace for gender and age. [Click to download!](https://www.kaggle.com/datasets/jangedoo/utkface-new)
*Note: You can experiment with any dataset you want.*

## Results:
Here are the screenshots of our project!
If you like this job, please help me and my friend by giving some stars.