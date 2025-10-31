# SurgNav-AR
🧠 Augmented Reality for Neurosurgical Navigation
3D Brain Lesion Segmentation (SAM) + AR Phantom Overlay (OpenCV)

This repository contains the implementation for a dual-pipeline neurosurgical navigation prototype integrating:

Automated lesion segmentation using a fine-tuned Segment Anything Model (SAM)

Marker-based Augmented Reality (AR) overlay using OpenCV and ArUco markers on a 3D-printed head phantom

This work was completed as part of my BSc Electrical & Computer Engineering degree at the University of Cape Town.

📌 Features

Fine-tuned SAM decoder for brain MRI segmentation

MRI → Volume → 3D mesh conversion pipeline

Camera calibration (Zhang method)

ArUco marker-based pose estimation

3D phantom overlay (OBJ models)

Pixel-RMSE overlay accuracy evaluation tool

📂 Repository Structure
📦 AR-Neurosurgery-Navigation
├── ARNavigationSAMOutputs/
├── ARResults/
├── ArUcoGeneratedMarkers/
├── BrainDataset/
├── CalibrationImages15/
├── ARNavigationSAMTraining.ipynb
├── ARTargetCalculations.py
├── CameraImageCalibration.py
├── GenerateMarker.py
├── MarkerBasedAROverlay.py
├── lesion.obj
├── skull.obj
└── README.md

🧬 Pipeline Overview
1️⃣ Lesion Segmentation (SAM)

Fine-tune SAM mask decoder only

Input: MRI slices + bounding-box prompts

5-fold cross-validation

Metrics: Dice, HD95, ASSD, volume accuracy

Performance (Median):

Metric	Score
Dice	82.66%
ASSD	1.49 mm
2️⃣ AR Navigation System

Camera calibration using chessboard pattern

Pose estimation via ArUco markers

OBJ lesion + skull overlay into real images

Manual refinement included

RMSE overlay evaluation tool

Accuracy:

Stage	Average Overlay Error
Before Refinement	>100 px
After Refinement	~5 px
🛠️ Installation
pip install torch torchvision torchaudio
pip install opencv-python numpy nibabel SimpleITK matplotlib
pip install transformers datasets

🚀 Usage
Train / Fine-Tune SAM

Run in Google Colab:

ARNavigationSAMTraining.ipynb

Calibrate Camera
python CameraImageCalibration.py

Generate ArUco Markers
python GenerateMarker.py

Run AR Overlay
python MarkerBasedAROverlay.py

Compute Pixel-RMSE Overlay Accuracy
python ARTargetCalculations.py

📊 Results Summary

SAM produced high-quality segmentation with fast training (2 epochs)

Accurate ArUco tracking & sub-pixel reprojection error

Manual refinement yielded precise alignment on phantom (~5 px RMSE)

📎 Dataset

Dataset used (not included in repo):
Head model dataset for mixed reality navigation in neurosurgical interventions for intracranial lesions (FigShare)

Includes MRI, segmentation masks, STL phantom, OBJ hologram files.

📜 Citation

If referencing this work:

Daya, S. (2025). Augmented Reality in Surgical Navigation of Brain Lesions using
Segment Anything Model and OpenCV. University of Cape Town.

👤 Author

Sachin Daya
BSc Electrical & Computer Engineering
University of Cape Town
Supervisor: Prof. Fred Nicolls

⭐ Future Work

LoRA / adapter-based SAM tuning

Automatic ICP refinement (no manual adjustment)

Real-time AR video feed

HoloLens / ARKit deployment

Usability testing with clinicians
