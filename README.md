PPE Detection System — YOLOv8

Computer vision system for real-time verification of Personal Protective Equipment (PPE) usage in industrial environments.

Overview

This project implements an automated PPE compliance checker using YOLOv8 object detection. The system detects whether workers are wearing required safety equipment (helmet, vest, boots, gloves) and generates visual alerts and CSV reports for non-compliance cases.

Results
Metric	Value
Precision	94.22%
Recall (Sensitivity)	98.21%
F1-Score	96.17%
Features
Real-time camera monitoring — continuous PPE verification from live camera feed
Image analysis — verify PPE compliance from local images
Quick capture — single-frame capture and verification
Automated reporting — generates reporte_EPP.csv with non-compliance details
GUI Interface — user-friendly Tkinter interface
Tech Stack
Python 3.x
YOLOv8 (Ultralytics)
OpenCV
Tkinter
Pandas
Scikit-learn
Dataset

Custom dataset of 2,706 annotated images from Roboflow with 5 classes:

helmet (casco)
vest (chaleco)
boots (botas)
gloves (guantes)
human (persona)

Dataset available at: https://drive.google.com/file/d/1wse3ektkYrQY0bX0EGEz30gse59R40B7/view?usp=drive_link

Installation
bash
pip install -r requirements.txt
Usage
Download the dataset and train the model (or use pretrained weights):
python
# Uncomment entrenar_modelo() in main.py to train
Run the application:
bash
python main.py
Select operation mode from the GUI:
Verificar desde Cámara — real-time camera verification
Verificar Imagen — load and verify a local image
Capturar Foto y Verificar — quick capture from camera
Model Training
Base model: YOLOv8n (pretrained on COCO)
Epochs: 40 with early stopping (patience=10)
Image size: 480×480
Data augmentation applied to underrepresented classes (gloves)
Academic Context

Developed as part of a Computer Vision course project at Universidad ESAN, Lima, Peru (2025).

Authors:

Diego Alberto Chavez Polinar
Gustavo Anderson Mendoza Montes
