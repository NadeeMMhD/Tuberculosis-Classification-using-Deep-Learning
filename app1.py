from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# MobileNet Model
class MobileNetModel(nn.Module):
    def __init__(self):
        super(MobileNetModel, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)

    def forward(self, x):
        return self.model(x)

# ResNet Model
class ResNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# EfficientNet Model
class EfficientNetModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0'):
        super(EfficientNetModel, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
models_dict = {
    'mobilenet': MobileNetModel().to(device),
    'resnet': ResNetModel().to(device),
    'efficientnet': EfficientNetModel().to(device)
}

# Load model weights
try:
    models_dict['mobilenet'].load_state_dict(torch.load('models/model_mob.pth', map_location=device))
    models_dict['resnet'].load_state_dict(torch.load('models/best_resnet_model.pth', map_location=device))
    models_dict['efficientnet'].load_state_dict(torch.load('models/best_efficientnet_model.pth', map_location=device))
except Exception as e:
    print(f"Error loading model weights: {e}")

# Set all models to evaluation mode
for model in models_dict.values():
    model.eval()

def predict_image(image_bytes, model_name):
    """
    Make a prediction on an image using the specified model
    """
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models_dict.keys())}")
    
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = models_dict[model_name](image)
        _, predicted = outputs.max(1)
        probability = torch.nn.functional.softmax(outputs, dim=1)[0]

    return predicted.item(), probability[1].item() * 100

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    model_name = request.form.get('model', 'mobilenet')  # Default to mobilenet if not specified

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        img_bytes = file.read()
        class_id, probability = predict_image(img_bytes, model_name)
        prediction = 'Tuberculosis' if class_id == 1 else 'Normal'

        return jsonify({
            'prediction': prediction,
            'probability': f'{probability:.2f}%',
            'model_used': model_name,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)