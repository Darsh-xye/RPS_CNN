# RPS_CNN
Real-time Rock-Paper-Scissors classifier using CNNs in PyTorch with webcam support via OpenCV.

This project is a complete computer vision pipeline for classifying Rock, Paper, and Scissors hand gestures using a custom Convolutional Neural Network (CNN) built with PyTorch.
It uses the TensorFlow Datasets (TFDS) Rock-Paper-Scissors dataset, applies powerful data augmentation, and supports:

✅ Custom Image Prediction

🎥 Live Webcam Inference using OpenCV

📉 Early Stopping

📦 Model Saving & Loading

📊 Dataset
Source: TensorFlow Datasets: rock_paper_scissors

Format: RGB images of hand gestures

Classes: rock, paper, scissors

Split: ~2,500 training samples, ~370 test samples

## 🧠 Model Architecture
A simple but effective CNN:

Input: 3×300×300  
↓ Conv2D(32) + BatchNorm + ReLU + MaxPool  
↓ Conv2D(64) + BatchNorm + ReLU + MaxPool  
↓ Conv2D(128) + BatchNorm + ReLU + MaxPool  
↓ Flatten → Linear(256) → ReLU → Dropout  
↓ Linear(3)


## 🧪 Performance

Metric	Value

Train Accuracy	~95%

Test Accuracy	~90–92%

Webcam	Real-time ✅

## 📦 Requirements

torch  
torchvision  
tensorflow-datasets  
numpy  
matplotlib  
Pillow  
opencv-python


## 💡 Features

Custom PyTorch data loader for TFDS

CNN with 3 conv blocks and dropout

Early stopping to prevent overfitting

Static image prediction

Live webcam gesture classification using OpenCV


## 👨‍💻 Author

Made by Darsh-xye
