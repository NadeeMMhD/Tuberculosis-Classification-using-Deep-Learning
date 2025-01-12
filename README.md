# Tuberculosis Classification with Deep Learning

This project uses deep learning models to classify chest X-rays as either **Tuberculosis** or **Normal**. The application supports MobileNetV2, ResNet50, and EfficientNet for binary classification.

## Features

- **Web Interface**: A Flask-based application for uploading X-ray images and viewing predictions.
- **Pretrained Models**: Fine-tuned MobileNetV2, ResNet50, and EfficientNet models for TB classification.
- **Training Scripts**: Modular scripts to train each model with support for early stopping and performance monitoring.
- **Visualization**: Loss and accuracy plots to track model training and validation.

---

## File Structure

### 1. `app1.py`
- Flask web app to serve model predictions.
- Endpoints:
  - `/`: Renders the homepage.
  - `/predict`: Accepts an X-ray image and predicts TB or normal lungs.
- Usage:
  - Place trained model weights (`*.pth` files) in a `models/` directory.

### 2. `effi.py`
- Training script for EfficientNet.
- Key Features:
  - Data augmentation.
  - Early stopping and model checkpointing.
  - Plots training/validation metrics.

### 3. `mob.py`
- Training script for MobileNetV2.
- Key Features:
  - Fine-tuning MobileNetV2 for binary classification.
  - Similar structure to `effi.py`.

### 4. `resnet.py`
- Training script for ResNet50.
- Key Features:
  - Freezes early ResNet layers to focus on the final classifier.
  - Saves the best model during training.

---

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/NadeeMMhD/Tuberculosis-Detection-using-Deep-Learning.git
    ```
    ```
    cd tb-classification
   
2. Install dependencies
``` 
   pip install -r requirements.txt
   ```
3. Download the pretrained model weights and place them in a models/ directory:
   models/model_mob.pth
   models/best_resnet_model.pth
   models/best_efficientnet_model.pth

4. Ensure dataset directories are structured as follows:
   TB_Chest_Radiography_Database/
   
    ├── Train/
   
    ├── Val/
   
    └── Test/

## Usage
1. Run the Flask App
```
   python app1.py
```
3.Train Models
  ``` 
 EfficientNet:

    python effi.py
```
 MobileNetV2:
 ```
  python mob.py
  ```
 ResNet50:
 ```
  python resnet.py
```
## Dependencies

Python 3.8+

PyTorch

torchvision

Flask

Matplotlib

PIL

Install all dependencies using:

    pip install -r requirements.txt





 
 

