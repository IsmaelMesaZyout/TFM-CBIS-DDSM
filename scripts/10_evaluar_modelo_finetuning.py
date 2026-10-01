"""
Script 10_evaluar_modelo_finetuning.py

Objetivo:
Evaluar el modelo entrenado con fine-tuning parcial de ResNet18 sobre el
conjunto de test del dataset CBIS-DDSM.

Qué hace este script:
1. Carga el modelo fine-tuned guardado.
2. Carga las imágenes ROI del conjunto de test.
3. Calcula accuracy, precision, recall y F1-score.
4. Genera la matriz de confusión.
5. Permite comparar este modelo con la baseline y con el modelo mejorado
   anterior basado únicamente en pesos de clase.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "processed" / "classification_roi"
MODEL_PATH = BASE_DIR / "models" / "resnet18_roi_finetuning_cpu.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo utilizado:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print("Clases:", test_dataset.classes)
print("Imágenes de test:", len(test_dataset))

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nAccuracy:")
print(accuracy_score(y_true, y_pred))

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=test_dataset.classes
))

print("\nMatriz de confusión:")
print(confusion_matrix(y_true, y_pred))