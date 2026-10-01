"""
Script 12_preparar_dataset_deteccion_completo.py

Objetivo:
Construir el dataset completo de detección sobre mamografía completa a partir de
CBIS-DDSM, utilizando una única clase general de lesión.

Qué hace este script:
1. Lee los CSV de masas y calcificaciones (train y test).
2. Localiza el DICOM de la mamografía completa a partir de 'image file path'.
3. Localiza el DICOM de la máscara ROI a partir de 'ROI mask file path'.
4. Identifica correctamente cuál de los DICOM corresponde a una máscara binaria.
5. Calcula la bounding box mínima que encierra la lesión.
6. Convierte la mamografía completa a PNG.
7. Guarda las etiquetas en formato YOLO:
      class_id x_center y_center width height
   con valores normalizados entre 0 y 1.
8. Si una misma mamografía contiene varias lesiones, las agrupa dentro del
   mismo archivo de etiquetas.

Formato de salida:
processed/detection_one_class/
├── images/
│   ├── train/
│   └── test/
└── labels/
    ├── train/
    └── test/

Clase utilizada:
0 = lesion

Importante:
Esta es la versión completa del script. A diferencia de la versión rápida,
no limita el número de filas procesadas.
"""

from pathlib import Path
import pandas as pd
import pydicom
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_DIR = BASE_DIR / "csv"
DICOM_DIR = BASE_DIR / "raw" / "dicom"
OUT_DIR = BASE_DIR / "processed" / "detection_one_class"

CSV_FILES = [
    ("mass_case_description_train_set.csv", "train"),
    ("calc_case_description_train_set.csv", "train"),
    ("mass_case_description_test_set.csv", "test"),
    ("calc_case_description_test_set.csv", "test"),
]


def normalize_to_uint8(img):
    img = img.astype(np.float32)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = img * 255.0
    return img.astype(np.uint8)


def dicom_to_png(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    img = normalize_to_uint8(img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(output_path)

    return img.shape[1], img.shape[0]  # width, height


def find_case_folder(csv_path):
    folder_name = str(csv_path).replace("\\", "/").split("/")[0]
    matches = list(DICOM_DIR.rglob(folder_name))

    if len(matches) == 0:
        return None

    return matches[0]


def is_mask_like(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    unique_values = len(np.unique(img))
    return unique_values <= 5


def find_full_mammogram_dicom(full_image_csv_path):
    case_folder = find_case_folder(full_image_csv_path)

    if case_folder is None:
        return None

    candidates = []

    for p in case_folder.rglob("*.dcm"):
        if "full mammogram images" in str(p):
            candidates.append(p)

    if len(candidates) == 0:
        return None

    return candidates[0]


def find_roi_mask_dicom(mask_csv_path):
    case_folder = find_case_folder(mask_csv_path)

    if case_folder is None:
        return None

    mask_candidates = []

    for p in case_folder.rglob("*.dcm"):
        if "ROI mask images" in str(p):
            mask_candidates.append(p)

    for dcm in mask_candidates:
        try:
            if is_mask_like(dcm):
                return dcm
        except Exception:
            pass

    all_dicoms = list(case_folder.rglob("*.dcm"))

    for dcm in all_dicoms:
        try:
            if is_mask_like(dcm):
                return dcm
        except Exception:
            pass

    return None


def get_bbox_from_mask(mask_dicom_path):
    ds = pydicom.dcmread(mask_dicom_path)
    mask = ds.pixel_array

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())

    return x_min, y_min, x_max, y_max


def bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    x_center = ((x_min + x_max) / 2.0) / img_width
    y_center = ((y_min + y_max) / 2.0) / img_height
    width = (x_max - x_min + 1) / img_width
    height = (y_max - y_min + 1) / img_height

    return x_center, y_center, width, height


dataset_entries = {
    "train": {},
    "test": {}
}

total_rows = 0
total_ok = 0
missing_full = 0
missing_mask = 0
empty_bbox = 0

for csv_name, split in CSV_FILES:
    print("\nProcesando:", csv_name)

    df = pd.read_csv(CSV_DIR / csv_name)

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Procesadas {idx} filas en {csv_name}...")

        total_rows += 1

        full_image_csv_path = row["image file path"]
        roi_mask_csv_path = row["ROI mask file path"]

        full_case_name = str(full_image_csv_path).replace("\\", "/").split("/")[0]
        output_stem = full_case_name

        full_dicom = find_full_mammogram_dicom(full_image_csv_path)
        mask_dicom = find_roi_mask_dicom(roi_mask_csv_path)

        if full_dicom is None:
            missing_full += 1
            continue

        if mask_dicom is None:
            missing_mask += 1
            continue

        bbox = get_bbox_from_mask(mask_dicom)

        if bbox is None:
            empty_bbox += 1
            continue

        if output_stem not in dataset_entries[split]:
            dataset_entries[split][output_stem] = {
                "full_dicom": full_dicom,
                "boxes": []
            }

        dataset_entries[split][output_stem]["boxes"].append(bbox)
        total_ok += 1

print("\nResumen de preparación:")
print("Filas totales:", total_rows)
print("Filas válidas con bbox:", total_ok)
print("Sin mamografía completa:", missing_full)
print("Sin máscara:", missing_mask)
print("Máscaras vacías:", empty_bbox)

for split in ["train", "test"]:
    images_dir = OUT_DIR / "images" / split
    labels_dir = OUT_DIR / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for stem, item in dataset_entries[split].items():
        image_out = images_dir / f"{stem}.png"
        label_out = labels_dir / f"{stem}.txt"

        img_width, img_height = dicom_to_png(item["full_dicom"], image_out)

        yolo_lines = []

        for bbox in item["boxes"]:
            x_min, y_min, x_max, y_max = bbox
            x_center, y_center, width, height = bbox_to_yolo(
                x_min, y_min, x_max, y_max, img_width, img_height
            )

            line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(line)

        with open(label_out, "w", encoding="utf-8") as f:
            for line in yolo_lines:
                f.write(line + "\n")

print("\nDataset de detección generado en:")
print(OUT_DIR)
print("\nEstructura esperada:")
print(OUT_DIR / "images" / "train")
print(OUT_DIR / "images" / "test")
print(OUT_DIR / "labels" / "train")
print(OUT_DIR / "labels" / "test")