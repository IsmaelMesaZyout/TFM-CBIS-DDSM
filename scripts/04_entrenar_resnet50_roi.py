#pip install torch torchvision scikit-learn matplotlib

"""
Script 04_entrenar_resnet50_roi.py

Objetivo:
Entrenar una primera baseline de clasificación binaria benigno/maligno
sobre las imágenes ROI reales extraídas del dataset CBIS-DDSM.

Entrada:
- processed/classification_roi/train/benign
- processed/classification_roi/train/malignant
- processed/classification_roi/test/benign
- processed/classification_roi/test/malignant

Qué hace este script:
1. Carga las imágenes ROI ya preprocesadas.
2. Redimensiona las imágenes a 224x224 píxeles.
3. Convierte las imágenes en escala de grises a 3 canales para poder usarlas
   con un modelo preentrenado en ImageNet.
4. Aplica normalización estándar compatible con modelos preentrenados.
5. Carga una red ResNet18 preentrenada mediante transfer learning.
6. Congela las capas convolucionales del modelo.
7. Sustituye la última capa para clasificar entre dos clases:
   benign y malignant.
8. Entrena únicamente la capa final durante pocas épocas, ya que se trata
   de una baseline inicial ejecutada en CPU.
9. Guarda el modelo entrenado en la carpeta models.

Nota:
Se utiliza ResNet18 en lugar de ResNet50 porque el entrenamiento se realizará
en CPU. ResNet18 es más ligera y permite validar el pipeline experimental
sin un coste computacional excesivo.
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
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


print("Clases:", train_dataset.classes)
print("Número de imágenes de entrenamiento:", len(train_dataset))
print("Número de imágenes de test:", len(test_dataset))


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


for param in model.parameters():
    param.requires_grad = False


model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)


epochs = 3


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


torch.save(model.state_dict(), MODEL_DIR / "resnet18_roi_baseline_cpu.pth")

print("Entrenamiento finalizado.")
print("Modelo guardado en:", MODEL_DIR / "resnet18_roi_baseline_cpu.pth")