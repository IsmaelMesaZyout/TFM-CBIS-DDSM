"""
Script 13_comprobar_dataset_deteccion.py

Objetivo:
Comprobar visualmente que las bounding boxes generadas en el dataset de
detección caen correctamente sobre las lesiones en las mamografías completas.

Qué hace este script:
1. Lee imágenes del dataset generado en la versión rápida.
2. Lee sus etiquetas en formato YOLO.
3. Convierte las coordenadas YOLO a coordenadas reales de píxel.
4. Dibuja las bounding boxes sobre la imagen.
5. Muestra varios ejemplos aleatorios.

Importante:
Este paso es fundamental antes de entrenar un detector, ya que permite validar
que las anotaciones son correctas.
"""

from pathlib import Path
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "processed" / "detection_one_class"

IMAGES_DIRS = [
    DATA_DIR / "images" / "train",
    DATA_DIR / "images" / "test"
]

LABELS_DIRS = {
    "train": DATA_DIR / "labels" / "train",
    "test": DATA_DIR / "labels" / "test"
}


def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """
    Convierte una bbox YOLO normalizada a coordenadas reales:
    x_min, y_min, x_max, y_max
    """
    x_center = x_center * img_width
    y_center = y_center * img_height
    width = width * img_width
    height = height * img_height

    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    return x_min, y_min, x_max, y_max


all_images = []

for folder in IMAGES_DIRS:
    split = folder.name
    for img_path in folder.glob("*.png"):
        all_images.append((img_path, split))

print("Número total de imágenes encontradas:", len(all_images))

sample_size = min(12, len(all_images))
sample = random.sample(all_images, sample_size)

plt.figure(figsize=(15, 10))

for i, (img_path, split) in enumerate(sample):
    label_path = LABELS_DIRS[split] / f"{img_path.stem}.txt"

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    img_width, img_height = img.size

    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id, x_center, y_center, width, height = parts
            x_center = float(x_center)
            y_center = float(y_center)
            width = float(width)
            height = float(height)

            x_min, y_min, x_max, y_max = yolo_to_bbox(
                x_center, y_center, width, height, img_width, img_height
            )

            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

    plt.subplot(3, 4, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"{split} - {img_path.stem}")
    plt.axis("off")

plt.tight_layout()
plt.show()