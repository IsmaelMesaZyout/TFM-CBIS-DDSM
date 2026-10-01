from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "processed" / "classification_roi"

for split in ["train", "test"]:
    print(f"\n{split.upper()}")
    for label in ["benign", "malignant"]:
        folder = DATA_DIR / split / label
        files = list(folder.glob("*.png"))
        print(label, len(files))

# mostrar ejemplos
all_images = list(DATA_DIR.rglob("*.png"))
#sample = random.sample(all_images, 6)
sample = random.sample(all_images, 20)

plt.figure(figsize=(12, 6))

for i, img_path in enumerate(sample):
    img = Image.open(img_path)
    #plt.subplot(2, 3, i + 1)
    plt.subplot(4, 5, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(img_path.parent.name)
    plt.axis("off")

plt.tight_layout()
plt.show()