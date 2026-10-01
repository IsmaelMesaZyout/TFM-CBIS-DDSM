from pathlib import Path
import pandas as pd
import pydicom
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]

CSV_DIR = BASE_DIR / "csv"
DICOM_DIR = BASE_DIR / "raw" / "dicom"
OUT_DIR = BASE_DIR / "processed" / "classification_roi"

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
    img = img * 255
    return img.astype(np.uint8)


def dicom_to_png(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    img = normalize_to_uint8(img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(output_path)

######################################### con este codigo salen imagenes negras, abajo a la derecha aparece 
# una máscara negra con una zona blanca. Eso NO debe entrar como ROI de clasificación
#def find_dicom_from_csv_path(csv_path):
#    folder_name = str(csv_path).split("/")[0]

    # Buscar en todo el dataset
#    matches = list(DICOM_DIR.rglob(folder_name))

#    if len(matches) == 0:
#        return None

    # coger la primera carpeta que coincida
#    folder = matches[0]

#   dicoms = list(folder.rglob("*.dcm"))
#
#    if len(dicoms) == 0:
#        return None
#
#    return dicoms[0]

#def find_dicom_from_csv_path(csv_path):
#    parts = str(csv_path).replace("\\", "/").split("/")
#
#    folder_name = parts[0]
#    series_id = parts[1] if len(parts) > 1 else None
#
#    folder_matches = list(DICOM_DIR.rglob(folder_name))
#
#    if len(folder_matches) == 0:
#        return None
#
#    folder = folder_matches[0]
#
#    if series_id:
#        series_matches = list(folder.rglob(series_id))
#        if len(series_matches) > 0:
#            folder = series_matches[0]
#
#    dicoms = list(folder.rglob("*.dcm"))
#
#    if len(dicoms) == 0:
#        dicoms = [p for p in folder.rglob("*") if p.is_file()]
#
#    if len(dicoms) == 0:
#        return None
#
#    return dicoms[0]

def is_mask_like(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array

    unique_values = len(np.unique(img))

    return unique_values <= 5


def find_dicom_from_csv_path(csv_path):
    parts = str(csv_path).replace("\\", "/").split("/")
    folder_name = parts[0]

    folder_matches = list(DICOM_DIR.rglob(folder_name))

    if len(folder_matches) == 0:
        return None

    base_folder = folder_matches[0]
    dicoms = list(base_folder.rglob("*.dcm"))

    if len(dicoms) == 0:
        return None

    non_masks = []

    for dcm in dicoms:
        try:
            if not is_mask_like(dcm):
                non_masks.append(dcm)
        except:
            pass

    if len(non_masks) == 0:
        return None

    return non_masks[0]

total_ok = 0
total_error = 0

for csv_name, split in CSV_FILES:
    df = pd.read_csv(CSV_DIR / csv_name)

    for idx, row in df.iterrows():
        pathology = row["pathology"]

        if pathology == "MALIGNANT":
            label = "malignant"
        elif pathology in ["BENIGN", "BENIGN_WITHOUT_CALLBACK"]:
            label = "benign"
        else:
            continue

        dicom_path = find_dicom_from_csv_path(row["cropped image file path"])

        if dicom_path is None:
            total_error += 1
            continue

        output_name = f"{csv_name.replace('.csv', '')}_{idx}_{row['patient_id']}.png"
        output_path = OUT_DIR / split / label / output_name

        try:
            dicom_to_png(dicom_path, output_path)
            total_ok += 1
        except Exception as e:
            print("Error:", dicom_path, e)
            total_error += 1

print("Imágenes convertidas:", total_ok)
print("Errores:", total_error)
print("Salida:", OUT_DIR)