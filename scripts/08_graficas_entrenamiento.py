"""
Script 08_graficas_entrenamiento.py

Objetivo:
Generar las gráficas principales del experimento para incluirlas en el TFM.

Qué hace este script:
1. Genera la gráfica de evolución de la pérdida (loss).
2. Genera la gráfica de evolución de la accuracy.
3. Genera la matriz de confusión del modelo mejorado.
4. Genera una comparación gráfica entre el modelo baseline y el modelo mejorado.

Las imágenes se guardan en la carpeta models.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# =========================
# 1. VALORES DE ENTRENAMIENTO
# =========================

train_losses = [0.6784, 0.6579, 0.6459, 0.6398, 0.6385]
train_accuracies = [0.5758, 0.6128, 0.6180, 0.6278, 0.6330]
epochs = range(1, len(train_losses) + 1)

# =========================
# 2. GRÁFICA DE LOSS
# =========================

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker="o")
plt.title("Evolución de la pérdida (Loss)")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.grid(True)
plt.tight_layout()
plt.savefig(MODEL_DIR / "loss.png", dpi=300)
plt.show()

# =========================
# 3. GRÁFICA DE ACCURACY
# =========================

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_accuracies, marker="o")
plt.title("Evolución de la accuracy")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.xticks(epochs)
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig(MODEL_DIR / "accuracy.png", dpi=300)
plt.show()

# =========================
# 4. MATRIZ DE CONFUSIÓN MODELO MEJORADO
# =========================

conf_matrix = np.array([
    [244, 184],
    [108, 168]
])

labels = ["Benign", "Malignant"]

plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix)
plt.title("Matriz de confusión - Modelo mejorado")
plt.xlabel("Clase predicha")
plt.ylabel("Clase real")
plt.xticks(ticks=np.arange(len(labels)), labels=labels)
plt.yticks(ticks=np.arange(len(labels)), labels=labels)
plt.colorbar()

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(
            j,
            i,
            conf_matrix[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig(MODEL_DIR / "matriz_confusion_mejorado.png", dpi=300)
plt.show()

# =========================
# 5. COMPARACIÓN BASELINE VS MEJORADO
# =========================

metricas = ["Accuracy", "Recall malignant", "Falsos negativos"]

baseline = [0.5909, 0.38, 170]
mejorado = [0.5852, 0.61, 108]

x = np.arange(len(metricas))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x - width / 2, baseline, width, label="Baseline")
plt.bar(x + width / 2, mejorado, width, label="Modelo mejorado")

plt.title("Comparación entre baseline y modelo mejorado")
plt.xlabel("Métrica")
plt.ylabel("Valor")
plt.xticks(x, metricas)
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(MODEL_DIR / "comparacion_baseline_mejorado.png", dpi=300)
plt.show()

print("Gráficas guardadas en:", MODEL_DIR)
print("- loss.png")
print("- accuracy.png")
print("- matriz_confusion_mejorado.png")
print("- comparacion_baseline_mejorado.png")