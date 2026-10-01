"""
Script 06_entrenar_resnet18_mejorado.py

Objetivo:
Entrenar una segunda versión del modelo de clasificación benigno/maligno
sobre ROI de CBIS-DDSM, mejorando la baseline inicial mediante el uso de
pesos de clase.

Motivación:
En la baseline anterior, el modelo obtuvo un recall bajo para la clase
"malignant", clasificando muchos casos malignos como benignos. En un contexto
médico, este error es especialmente crítico, ya que equivale a falsos negativos.

Qué hace este script:
1. Carga el dataset ROI procesado.
2. Usa ResNet18 preentrenada mediante transfer learning.
3. Congela las capas convolucionales.
4. Sustituye la capa final para clasificar entre benign y malignant.
5. Aplica pesos de clase en la función de pérdida para penalizar más los errores
   en la clase malignant.
6. Entrena el modelo durante varias épocas.
7. Guarda el nuevo modelo mejorado para evaluarlo posteriormente.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "processed" / "classification_roi"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo utilizado:", device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

print("Clases:", train_dataset.classes)
print("Número de imágenes de entrenamiento:", len(train_dataset))


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)


# Clases: ['benign', 'malignant']
# Se asigna más peso a la clase malignant para reducir falsos negativos.
class_weights = torch.tensor([1.0, 1.4], dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

epochs = 5


for epoch in range(epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Loss: {epoch_loss:.4f} "
        f"- Accuracy: {epoch_acc:.4f}"
    )


torch.save(model.state_dict(), MODEL_DIR / "resnet18_roi_weighted_cpu.pth")

print("Entrenamiento mejorado finalizado.")
print("Modelo guardado en:", MODEL_DIR / "resnet18_roi_weighted_cpu.pth")