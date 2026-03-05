# kaggle_Multiview_Pig_Posture_Recognition

<img src="pictures/rank.jpg">

I ranked **28th** in the [Kaggle Multi-view Pig Posture Recognition competition](https://www.kaggle.com/competitions/multi-view-pig-posture-recognition)！

## Introduction

The goal of the competition is to develop computer vision models that can automatically recognize the posture of pigs from multi-view images in a farm environment. Each image contains annotated bounding boxes around pigs, and the task is to classify the posture of each pig into one of several categories such as standing, sitting, or lying.

Pig posture recognition plays an important role in precision livestock farming. Changes in posture and activity can reflect the health, welfare, and behavior of animals. In this project, we explore deep learning–based approaches for pig detection and posture classification, evaluate model performance on the competition dataset, and analyze the robustness of the models across different camera views and environments.

## Dataset

This project uses the dataset provided in the Kaggle competition  
[Multi-view Pig Posture Recognition](https://www.kaggle.com/competitions/multi-view-pig-posture-recognition/data).  
We use the **train2** subset released by the competition as the primary training source.

Each image contains bounding box annotations for pigs along with a posture label.  
The task is to detect pigs and classify their posture into one of the following categories:

| Class ID | Posture Name |
|---------|--------------|
| 0 | Lateral_lying_left |
| 1 | Lateral_lying_right |
| 2 | Sitting |
| 3 | Standing |
| 4 | Sternal_lying |

To prevent **data leakage**, images captured in the same scene and time window are grouped together.  
Since multiple frames may be captured from the same camera within a short time interval, we apply a **45-second temporal threshold** to prevent data leakage between the training and validation sets.
In addition, all images captured on **Feb 9th, 2025** are used as the test set to simulate a real deployment scenario. The model has never seen data from that day during training.

After grouping and splitting the dataset, the final dataset sizes are:

- **Train:** (16,678)  
- **Validation:** (4,512)  
- **Test:** (1,744)

## Exploratory Data Analysis

To better understand the dataset, we first examine the class distribution and camera-view distribution across the dataset.

### Class Distribution

![Class Distribution](pictures/class_train.jpg)

![Class Distribution (Validation)](pictures/class_val.jpg)

The distribution of posture classes is highly imbalanced. Certain postures such as Standing appear significantly more frequently than others. This imbalance reflects real-world farm environments, where pigs spend different amounts of time in different postures.

### Camera View Distribution

![Camera Distribution](pictures/cam_train.jpg)

![Camera Distribution (Test)](pictures/cam_test.jpg)

The dataset is collected from multiple camera angles, and the distribution of camera views is also not uniform. Some cameras contribute a much larger number of images than others.

To better simulate a real-world deployment scenario, the validation set is constructed to mimic the camera-view distribution and posture distribution observed in real farm data. By aligning the validation distribution with realistic conditions, we aim to obtain a more reliable estimate of model performance when deployed in practice.


