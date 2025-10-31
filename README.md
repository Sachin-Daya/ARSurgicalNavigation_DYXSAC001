# SurgNav-AR
# Augmented Reality in Surgical Navigation of Brain Lesions
### Using Segment Anything Model (SAM) and OpenCV

[cite_start]A final year undergraduate project (EEE4022S) by **Sachin Daya** (DYXSAC001) submitted to the Department of Electrical Engineering at the University of Cape Town[cite: 7, 10, 11].

[cite_start]**Supervisor:** Fred Nicolls [cite: 9]

---

## 🚀 Project Overview

[cite_start]This project designs, implements, and evaluates a two-part proof-of-concept pipeline for AR-assisted neurosurgical navigation[cite: 114]. The system addresses two primary bottlenecks in the field: 
1.  [cite_start]**Segmentation Inefficiency:** The time-consuming and variable nature of manually creating 3D models from preoperative scans[cite: 105].
2.  [cite_start]**Registration Accuracy:** The difficulty of accurately aligning the virtual 3D models with the physical patient's anatomy[cite: 107].

The pipeline is structured as follows:
1.  [cite_start]**Part 1 (Segmentation):** A fine-tuned Segment Anything Model (SAM) is used to automatically segment brain lesions from preoperative MRI scans[cite: 116].
2.  [cite_start]**Part 2 (AR Overlay):** An OpenCV-based, marker-driven AR system overlays the generated 3D lesion model onto a corresponding physical 3D-printed head phantom[cite: 117].


---

## 🔬 Project Components

### 🧠 Part 1: 3D Lesion Segmentation (SAM)

* [cite_start]**Objective:** To investigate the feasibility of using a fine-tuned foundation model (SAM) for efficient and accurate brain lesion segmentation[cite: 116].
* [cite_start]**Model:** `sam-vit-base` [cite: 477]
* [cite_start]**Methodology:** The mask decoder of a pre-trained SAM was fine-tuned on 15 MRI cases from a public neurosurgical dataset[cite: 47, 122, 378]. [cite_start]Bounding-box prompts, generated from the ground-truth masks, were used for training[cite: 122, 542].
* **Key Script:** `ARNavigationSAMTraining.ipynb`
* [cite_start]**Result:** The model showed promising efficiency, achieving a **median 3D Dice score of 82.66%** and a **median Average Symmetric Surface Distance (ASSD) of 1.49 mm** after only 2 epochs of training[cite: 50, 1191, 1192].

### 👻 Part 2: Augmented Reality Overlay (OpenCV)

* [cite_start]**Objective:** To develop and evaluate a marker-based AR system to visualize the 3D models on a physical phantom[cite: 117].
* **Methodology:** An ArUco marker-based system developed in OpenCV. [cite_start]The system detects a marker [cite: 117] [cite_start]to establish an initial 6-DoF pose and then renders the 3D models (`skull.obj`, `lesion.obj`) onto the camera's view of a physical 3D-printed phantom[cite: 435]. [cite_start]Manual refinement controls were included to improve alignment[cite: 49].
* **Key Scripts:** `CameraImageCalibration.py`, `MarkerBasedAROverlay.py`, `ARTargetCalculations.py`
* **Result:**
    * [cite_start]ArUco pose estimation was highly precise, with **sub-pixel reprojection error** (avg. < 0.9 px)[cite: 52, 1376].
    * [cite_start]Initial registration based *only* on the marker was inaccurate (RMSE > 100 pixels)[cite: 53, 1392].
    * [cite_start]After manual refinement, the final average alignment error was reduced to **5.0587 pixels**[cite: 53, 1434].

### **⚠️ Important Note on AR Evaluation**

[cite_start]The 3D-printed phantom was based on `case_08` from the dataset[cite: 787]. [cite_start]The segmentation pipeline's performance on this specific case was poor (Dice3D 57.95%), resulting in a fragmented 3D model[cite: 1278, 1311].

[cite_start]Therefore, to meaningfully evaluate the AR system's *registration* accuracy, the **ground truth lesion model** for `case_08` was used in the AR overlay, not the model predicted by the SAM pipeline [cite: 1346-1352].

---

## 📁 Repository Structure

* `ARNavigationSAMTraining.ipynb`: Jupyter Notebook for training and evaluating the SAM segmentation model.
* `MarkerBasedAROverlay.py`: Main Python script to run the AR overlay application (uses a static image).
* `ARTargetCalculations.py`: Helper script to calculate the final pixel alignment error using manual clicks.
* `CameraImageCalibration.py`: Script to perform camera calibration using a chessboard.
* `GenerateMarker.py`: Script to generate the ArUco markers used.
* `ARNavigationSAMOutputs/`: Contains sample outputs (3D meshes) from the segmentation pipeline.
* `ARResults/`: Contains "before" and "after" images from the AR overlay evaluation.
* [cite_start]`BrainDataset/`: Information on the public dataset used[cite: 5].
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

* [cite_start]This work was completed in partial fulfillment of the academic requirements for a Bachelor of Science degree in Electrical and Computer Engineering at the University of Cape Town[cite: 12].
* Guidance from supervisor Fred Nicolls.
* The public dataset: "Head model dataset for mixed reality navigation in neurosurgical interventions for intracranial lesions" by Qi et al. (2024) [cite_start][cite: 5].

