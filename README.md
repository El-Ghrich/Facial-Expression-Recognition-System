# Facial Expression Recognition

## Acknowledgment

This project was developed under the supervision of **LACHKAR ABDELMOUNAIM** as part of the **as part of the "Image Processing" module** at the **École Nationale des Sciences Appliquées de Tanger (ENSA Tanger)**.

Anyone using, reproducing, or adapting this GitHub repository is kindly requested to acknowledge its academic origin by citing the supervisor, the project context, and the institution in the description of their work.


A deep learning system that recognizes 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise) from facial images using a custom CNN built with TensorFlow and Keras. Includes offline evaluation and real-time webcam inference.

## Project Structure

| File | Purpose |
|------|---------|
| `data_loader.py` | Loads images, applies augmentation, creates generators |
| `model_train.py` | Builds and trains the CNN model |
| `evaluate_webcam.py` | Evaluates on test set + runs live webcam demo |
| `emotion_model.h5` | Trained model weights |
| `confusion_matrix.png` | Evaluation confusion matrix |
| `dataset/` | FER dataset (train/ + test/ folders) |
| `.venv310/` | Python 3.10 virtual environment (TF 2.10 + DirectML) |

## Setup

Activate the environment (required before running any script):

```powershell
.\.venv310\Scripts\Activate.ps1
```

## Usage

### Offline Evaluation
```powershell
python evaluate_webcam.py --mode eval
```
Prints classification report and saves `confusion_matrix.png`.

### Real-Time Webcam
```powershell
python evaluate_webcam.py
```
Press `q` in the camera window to quit.

### Train from Scratch
```powershell
python model_train.py
```
Trains for 50 epochs (or until early stop). Saves best weights to `emotion_model.h5`.

### Data Loader (standalone)
```powershell
python data_loader.py --path dataset --batch_size 64
```

## Requirements

- Python 3.10
- tensorflow 2.10.0 + tensorflow-directml-plugin
- opencv-python, scikit-learn, seaborn, matplotlib, pillow, scipy

All dependencies are pre-installed in `.venv310/`.
