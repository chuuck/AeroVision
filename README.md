# Aerial Image Classifiers (WIP) 🏞️

AeroVision is a project that explores building different image classifiers trained on Aerial Image Dataset.

### Introduction

This project explores the use of image classifiers on the Aerial Image Dataset. The goal is to build and evaluate multiple models, including a custom model and pre-trained models like ResNet, to achieve optimal image classification results.

The project aims to also explore different stages of ML model development:

1. Data Analysis & Data Cleaning
2. Model building/modification
3. Training
4. Evaluation
5. Inference

## About the Project

### Dataset Overview

| Class 1 | Class 2 | Class 3 | Class 4 |
| --- | --- | --- | --- |
| ![Class 1: Aerial View](./data/images/class1/image1.jpg) | ![Class 2: Building](./data/images/class2/image2.jpg) | ![Class 3: Road](./data/images/class3/image3.jpg) | ![Class 4: Tree](./data/images/class4/image4.jpg) |
| ![Class 5: Car](./data/images/class5/image5.jpg) | ![Class 6: Person](./data/images/class6/image6.jpg) | ![Class 7: Bike](./data/images/class7/image7.jpg) | ![Class 8: Bus](./data/images/class8/image8.jpg) |

### Dataset Analysis


### Building datasets

The `dataset_builder` notebook is used to create the CSV files required for training the model. Three types of datasets are currently supported:
- Whole Dataset: Uses every image from each class.
- Bluriness Threshold Dataset: Uses only images with a bluriness level above a specified threshold.
- Balanced Dataset: Balances the entire dataset and uses a specified number of images from each class.

To create a dataset, run the `dataset_builder` notebook and select the desired dataset type. The resulting CSV file will be used as input for the training process.

### Models

In this project, I aim to explore the performance of various deep learning models on the Aerial Image Dataset. Currently, I have implemented a custom convolutional neural network (CNN) model, which serves as a baseline for the experiments.

The future plan is to also implement ResNet50, VGG16 and InceptionV3. Their performances will be evaluated against each other and the results will be shown in the Results section of this project.

### Testing Methodology

To evaluate the performance of the model, I conducted a series of tests on the Aerial Image Dataset. The following metrics were measured:

- **Training and Testing Loss:** tracked the loss of the model during training and testing to ensure that it is converging and generalizing well to unseen data.
- **Training and Testing Accuracy:** measured the accuracy of the model on both the training and testing sets to evaluate its performance on seen and unseen data.
- **Classification Report:** generated a classification report for each class, which includes metrics such as:
  - True Positives (TP)
  - True Negatives (TN)
  - False Positives (FP)
  - False Negatives (FN)
  - F1-Score
- **Confusion Matrix:** created a confusion matrix to visualize the performance of the model on each class and identify any potential biases or errors.

## Instructions

### 1. Getting Started 🚀
    
1. Set up the environment for the project and install necessary dependencies:

   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. To see if everything works well, if no errors are observed you are good to proceed:

   ```
   python inference.py path-to-model path-to-image
   ```

### 2. Building the dataset 



### 3. Selecting the Model Architecture + Custom Architecture

### 4. Training the model

### 5. Evaluation

Most of the evaluation is performed after the model is done training. However, it is possibe to get classification report and confusion matrix after the model is saved and stored in checkpoints folder. Simply:

```
python eval.py datasets/testing.csv checkpoints/custom_model.pt
```

### 6. Inference

To run the inference use the inference.py script, model from the checkpoints filder with the image you want to test.

```
python inference.py checkpoint/custom_model.pt images/sample1.jpg
```

## Results

## Conclusions




