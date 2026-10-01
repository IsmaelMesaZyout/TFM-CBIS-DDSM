"""
Script 09_entrenar_resnet18_finetuning.py

Objetivo:
Realizar una última mejora del modelo de clasificación benigno/maligno
sobre ROI del dataset CBIS-DDSM mediante fine-tuning parcial de ResNet18.

Motivación:
En experimentos anteriores se entrenó únicamente la capa final del modelo,
manteniendo congeladas todas las capas convolucionales. Esta estrategia
permite construir una baseline rápida y estable, pero limita la capacidad
de adaptación del modelo al dominio mamográfico.

En este script se descongelan las últimas capas profundas de ResNet18
(layer4) junto con la capa fully connected final. De este modo, el modelo
puede reajustar parte de sus representaciones visuales al problema concreto
de clasificación de lesiones mamarias.

Qué hace este script:
1. Carga el dataset ROI procesado.
2. Aplica transformaciones y aumento de datos moderado.
3. Carga ResNet18 preentrenada en ImageNet.
4. Congela la mayor parte de la red.
5. Descongela layer4 y la capa final fc.
6. Usa pesos de clase para penalizar más los errores en la clase malignant.
7. Entrena el modelo durante varias épocas.
8. Guarda el modelo final y las gráficas de entrenamiento.

Nota:
Este experimento está pensado para ejecutarse en CPU, por lo que se mantiene
una configuración razonable en cuanto a batch size y número de épocas.
"""

from pathlib import Path
import matplotlib.pyplot as plt
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

# =========================
# TRANSFORMACIONES
# =========================

train_transform = transforms.Compose([
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

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASETS Y DATALOADERS
# =========================

train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print("Clases:", train_dataset.classes)
print("Número de imágenes de entrenamiento:", len(train_dataset))
print("Número de imágenes de test:", len(test_dataset))

# =========================
# MODELO
# =========================

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Congelar toda la red
for param in model.parameters():
    param.requires_grad = False

# Descongelar solo layer4
for param in model.layer4.parameters():
    param.requires_grad = True

# Sustituir la capa final
model.fc = nn.Linear(model.fc.in_features, 2)

# La fc también debe entrenarse
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

# =========================
# FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
# =========================

class_weights = torch.tensor([1.0, 1.4], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Solo se optimizan los parámetros entrenables
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)

epochs = 8

train_losses = []
train_accuracies = []

# =========================
# ENTRENAMIENTO
# =========================

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

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Loss: {epoch_loss:.4f} "
        f"- Accuracy: {epoch_acc:.4f}"
    )

# =========================
# GUARDAR MODELO
# =========================

model_path = MODEL_DIR / "resnet18_roi_finetuning_cpu.pth"
torch.save(model.state_dict(), model_path)

print("Entrenamiento finalizado.")
print("Modelo guardado en:", model_path)

# =========================
# GRÁFICAS DE ENTRENAMIENTO
# =========================

epochs_range = range(1, epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_losses, marker="o")
plt.title("Evolución de la pérdida (Fine-tuning parcial)")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.xticks(list(epochs_range))
plt.grid(True)
plt.tight_layout()
plt.savefig(MODEL_DIR / "loss_finetuning.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_accuracies, marker="o")
plt.title("Evolución de la accuracy (Fine-tuning parcial)")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.xticks(list(epochs_range))
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig(MODEL_DIR / "accuracy_finetuning.png", dpi=300)
plt.show()

print("Gráficas guardadas en:", MODEL_DIR)
print("- loss_finetuning.png")
print("- accuracy_finetuning.png")