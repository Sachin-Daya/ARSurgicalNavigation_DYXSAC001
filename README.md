# Augmented Reality in Surgical Navigation of Brain Lesions
### Using Segment Anything Model (SAM) and OpenCV

A final year undergraduate project (EEE4022S) by **Sachin Daya** (DYXSAC001) submitted to the Department of Electrical Engineering at the University of Cape Town.

**Supervisor:** Fred Nicolls

---

## 🚀 Project Overview

This project presents a proof-of-concept pipeline for Augmented Reality (AR) in surgical navigation, designed to address two major challenges in the field: inefficient 3D model creation and inaccurate virtual-to-physical registration.

The pipeline is divided into two main components:
1.  **Automated Segmentation:** This part investigates the use of a fine-tuned Segment Anything Model (SAM) for automatically segmenting brain lesions from preoperative MRI data. The mask decoder of a pre-trained SAM was fine-tuned on 15 MRI cases, demonstrating an efficient method for generating 3D models.
2.  **Marker-Based AR Overlay:** This part develops an AR system using OpenCV and ArUco markers. The system overlays the 3D lesion models onto a physical 3D-printed head phantom, allowing for visualization in a real-world context.

The project successfully demonstrates the feasibility of this combined approach. The segmentation component achieved promising accuracy, with a median 3D Dice score of 82.66%. The AR component showed that while initial marker-based registration was inaccurate (RMSE > 100 pixels), manual refinement controls allowed users to achieve a precise final alignment with an average error of 5.0587 pixels.



---

## 🔬 Project Components

### 🧠 Part 1: 3D Lesion Segmentation (SAM)

* **Objective:** To investigate the feasibility of using a fine-tuned foundation model (SAM) for efficient and accurate brain lesion segmentation.
* **Model:** `sam-vit-base`
* **Methodology:** The mask decoder of a pre-trained SAM was fine-tuned on 15 MRI cases from a public neurosurgical dataset. Bounding-box prompts, generated from the ground-truth masks, were used for training.
* **Key Script:** `ARNavigationSAMTraining.ipynb`
* **Result:** The model showed promising efficiency, achieving a **median 3D Dice score of 82.66%** and a **median Average Symmetric Surface Distance (ASSD) of 1.49 mm** after only 2 epochs of training.

### 👻 Part 2: Augmented Reality Overlay (OpenCV)

* **Objective:** To develop and evaluate a marker-based AR system to visualize the 3D models on a physical phantom.
* **Methodology:** An ArUco marker-based system developed in OpenCV. The system detects a marker to establish an initial 6-DoF pose and then renders the 3D models (`skull.obj`, `lesion.obj`) onto the camera's view of a physical 3D-printed phantom. Manual refinement controls were included to improve alignment.
* **Key Scripts:** `CameraImageCalibration.py`, `MarkerBasedAROverlay.py`, `ARTargetCalculations.py`
* **Result:**
    * ArUco pose estimation was highly precise, with **sub-pixel reprojection error** (avg. < 0.9 px).
    * Initial registration based *only* on the marker was inaccurate (RMSE > 100 pixels).
    * After manual refinement, the final average alignment error was reduced to **5.0587 pixels**.

### **⚠️ Important Note on AR Evaluation**

The 3D-printed phantom was based on `case_08` from the dataset. The segmentation pipeline's performance on this specific case was poor (Dice3D 57.95%), resulting in a fragmented 3D model.

Therefore, to meaningfully evaluate the AR system's *registration* accuracy, the **ground truth lesion model** for `case_08` was used in the AR overlay, not the model predicted by the SAM pipeline.

---

## 📁 Repository Structure

* `ARNavigationSAMTraining.ipynb`: Jupyter Notebook for training and evaluating the SAM segmentation model.
* `MarkerBasedAROverlay.py`: Main Python script to run the AR overlay application (uses a static image).
* `ARTargetCalculations.py`: Helper script to calculate the final pixel alignment error using manual clicks.
* `CameraImageCalibration.py`: Script to perform camera calibration using a chessboard.
* `GenerateMarker.py`: Script to generate the ArUco markers used.
* `ARNavigationSAMOutputs/`: Contains sample outputs (3D meshes) from the segmentation pipeline.
* `ARResults/`: Contains "before" and "after" images from the AR overlay evaluation.
* `BrainDataset/`: Information on the public dataset used.
* `CalibrationImages15/`: The 15 images used for the final camera calibration.
* `ArUcoGeneratedMarkers/`: Images of the markers used in the AR setup.
* `*.obj`: The 3D models (`skull.obj`, `lesion.obj`, `Markers.obj`) used for the AR overlay.
* `PrintChessboard.docx`: The chessboard pattern used for calibration.

---

## ⚙️ Setup and Usage

### 1. Segmentation (Part 1)
1.  Download the required dataset (details in `BrainDataset/`).
2.  Configure the data paths within `ARNavigationSAMTraining.ipynb`.
3.  Run the notebook cells to train the model and generate 3D mesh outputs. (A GPU environment like Google Colab is recommended).

### 2. AR Overlay (Part 2)
1.  **Install dependencies** (e.g., `opencv-python`, `numpy`, `trimesh`).
2.  **Calibrate your camera:**
    * Print `PrintChessboard.docx`.
    * Take multiple pictures of the chessboard (see `CalibrationImages15/` for examples).
    * Run `CameraImageCalibration.py` to generate your camera's calibration file (`.npz`).
3.  **Run the AR Overlay:**
    * Place the 3D-printed phantom and ArUco markers in view of the camera.
    * Update `MarkerBasedAROverlay.py` to point to your calibration file and a static test image.
    * Run `MarkerBasedAROverlay.py`. This will load the models, find the marker, and overlay the 3D model.
    * Use the keyboard controls (defined in the script) to manually refine the alignment.
4.  **Evaluate Alignment:**
    * Save a final "after" image from the overlay script.
    * Run `ARTargetCalculations.py` and click on corresponding physical and virtual landmarks to calculate the final RMSE error.

---

## Acknowledgments

* This work was completed in partial fulfillment of the academic requirements for a Bachelor of Science degree in Electrical and Computer Engineering at the University of Cape Town.
* Guidance from supervisor Fred Nicolls.
* The public dataset: "Head model dataset for mixed reality navigation in neurosurgical interventions for intracranial lesions" by Qi et al. (2024).

