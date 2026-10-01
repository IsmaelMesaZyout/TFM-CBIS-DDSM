"""
Evalúa el modelo entrenado sobre el conjunto de test.
Calcula accuracy, precision, recall, F1-score y matriz de confusión.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "processed" / "classification_roi"
MODEL_PATH = BASE_DIR / "models" / "resnet18_roi_baseline_cpu.pth"

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

# Conclusión:
# El modelo baseline basado en ResNet18 obtuvo una accuracy del 59,09% sobre el conjunto de test.
# En la clase "benign" alcanzó una precision de 0,65, un recall de 0,72 y un F1-score de 0,68.
# En la clase "malignant" obtuvo una precision de 0,47, un recall de 0,38 y un F1-score de 0,42.
#
# La matriz de confusión fue:
# [[310 118]
#  [170 106]]
#
# Esto significa que el modelo clasificó correctamente 310 casos benignos y 106 casos malignos.
# Sin embargo, 170 casos malignos fueron clasificados erróneamente como benignos, lo que supone
# un número elevado de falsos negativos. En un contexto médico, este aspecto es especialmente
# relevante, ya que interesa maximizar el recall de la clase maligna.
#
# Estos resultados constituyen una primera baseline válida, aunque todavía limitada. El siguiente
# paso será mejorar el modelo mediante pesos de clase, aumento de datos y/o un entrenamiento más prolongado.