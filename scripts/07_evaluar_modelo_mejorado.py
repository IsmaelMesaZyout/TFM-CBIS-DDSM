"""
Script 07_evaluar_modelo_mejorado.py

Objetivo:
Evaluar el modelo mejorado (con pesos de clase) sobre el conjunto de test
y compararlo con la baseline inicial.

Se calculan:
- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusión

Especial atención:
Mejorar el recall de la clase "malignant" y reducir falsos negativos.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "processed" / "classification_roi"
MODEL_PATH = BASE_DIR / "models" / "resnet18_roi_weighted_cpu.pth"

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

"""
El modelo mejorado mediante el uso de pesos de clase mantiene una accuracy similar a la baseline (≈58-59%), 
pero introduce una mejora significativa en la detección de casos malignos. En particular, el recall de la clase "malignant" aumenta de 0,38 a 0,61, 
reduciendo notablemente el número de falsos negativos (de 170 a 108 casos).

Este resultado pone de manifiesto el impacto del desbalanceo de clases en problemas de clasificación médica 
y demuestra que la incorporación de pesos en la función de pérdida permite priorizar la detección de la clase minoritaria.

Aunque esta mejora implica un incremento de falsos positivos en la clase benigna, 
desde el punto de vista clínico resulta preferible, ya que minimiza el riesgo de no detectar casos de cáncer. 
Por tanto, el modelo mejorado presenta un comportamiento más adecuado para aplicaciones médicas, donde el recall de la clase maligna es un factor crítico.
"""