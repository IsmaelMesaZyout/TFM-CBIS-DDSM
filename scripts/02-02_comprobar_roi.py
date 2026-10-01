#Durante la fase de preparación del conjunto de datos, se realizó una verificación específica 
#con el objetivo de asegurar que las imágenes seleccionadas correspondieran a regiones de interés (ROI) reales y no a máscaras de segmentación.
#
#Para ello, se implementó un script de análisis sobre los archivos DICOM asociados a las rutas 
#proporcionadas en los ficheros CSV. Dicho script permitía inspeccionar propiedades básicas de cada imagen, 
#tales como sus dimensiones, valores de intensidad y, especialmente, el número de valores únicos de píxel.
#
#A partir de este análisis se observó que las máscaras binarias presentan un número muy reducido de valores 
#únicos (por ejemplo, 2 valores), mientras que las imágenes ROI reales contienen una alta variabilidad de intensidades, 
#alcanzando valores del orden de miles (por ejemplo, 1190 valores únicos en algunos casos), reflejando las características del tejido mamario.
#
#En base a esta diferencia, se definió un criterio de filtrado automático que descarta aquellas imágenes 
#con un número de valores únicos inferior o igual a un umbral determinado, identificándolas como máscaras. 
#De este modo, únicamente se seleccionaron imágenes ROI reales para la construcción del dataset final.
#
#Este procedimiento garantiza que el modelo de clasificación benigno/maligno se entrene exclusivamente 
#con información radiológica relevante, evitando el aprendizaje de patrones artificiales derivados de anotaciones binarias.

#def is_mask_like(dicom_path):
#    ds = pydicom.dcmread(dicom_path)
#    img = ds.pixel_array
#
#    unique_values = len(np.unique(img))
#
#    return unique_values <= 5


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

    if len(non_masks) > 0:
        return non_masks[0]

    return dicoms[0]

df = pd.read_csv(CSV_DIR / "mass_case_description_train_set.csv")

for i in range(5):
    csv_path = df.loc[i, "cropped image file path"]
    dcm = find_dicom_from_csv_path(csv_path)

    print("CSV:", csv_path)
    print("DICOM encontrado:", dcm)
    print()


df = pd.read_csv(CSV_DIR / "mass_case_description_train_set.csv")

csv_path = df.loc[0, "cropped image file path"]
folder_name = str(csv_path).replace("\\", "/").split("/")[0]

base_folder = list(DICOM_DIR.rglob(folder_name))[0]
dicoms = list(base_folder.rglob("*.dcm"))

print("Carpeta:", base_folder)
print("Número de DICOM:", len(dicoms))

for dcm in dicoms:
    ds = pydicom.dcmread(dcm)
    img = ds.pixel_array
    print(dcm)
    print("shape:", img.shape)
    print("min:", img.min(), "max:", img.max(), "unique:", len(np.unique(img)))
    print()