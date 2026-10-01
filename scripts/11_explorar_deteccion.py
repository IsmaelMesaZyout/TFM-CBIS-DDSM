"""
Script 11_explorar_deteccion.py

Objetivo:
Comprobar que se pueden localizar correctamente, para un mismo caso del dataset
CBIS-DDSM, la mamografía completa y la máscara ROI asociada.

Qué hace este script:
1. Lee los CSV principales del dataset.
2. Toma algunos ejemplos de mass y calc.
3. Busca en disco el DICOM correspondiente a:
   - image file path (mamografía completa)
   - ROI mask file path (máscara de la lesión)
4. Muestra por consola las rutas encontradas.

Importante:
En esta fase no se genera todavía el dataset de detección ni se calculan
bounding boxes. Solo se valida que el emparejamiento entre imagen completa
y máscara funciona correctamente.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_DIR = BASE_DIR / "csv"
DICOM_DIR = BASE_DIR / "raw" / "dicom"

CSV_FILES = [
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv",
    "calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv",
]


def find_case_folder(csv_path):
    """
    Localiza la carpeta principal del caso usando el primer fragmento de la ruta CSV.
    Ejemplo:
    Mass-Training_P_00001_LEFT_CC -> carpeta real dentro de raw/dicom
    """
    folder_name = str(csv_path).replace("\\", "/").split("/")[0]
    matches = list(DICOM_DIR.rglob(folder_name))

    if len(matches) == 0:
        return None

    return matches[0]


def find_any_dicom_under(csv_path):
    """
    Busca cualquier archivo DICOM dentro de la carpeta del caso.
    Se usará de momento para comprobar que el caso existe y contiene DICOM.
    """
    case_folder = find_case_folder(csv_path)

    if case_folder is None:
        return None

    dicoms = list(case_folder.rglob("*.dcm"))

    if len(dicoms) == 0:
        return None

    return dicoms[0]


for csv_name in CSV_FILES:
    print("\n" + "=" * 80)
    print("CSV:", csv_name)

    df = pd.read_csv(CSV_DIR / csv_name)

    # Mostrar solo 3 ejemplos por archivo para comprobar rápidamente
    for i in range(min(3, len(df))):
        row = df.iloc[i]

        full_image_path = row["image file path"]
        roi_mask_path = row["ROI mask file path"]

        full_dicom = find_any_dicom_under(full_image_path)
        mask_dicom = find_any_dicom_under(roi_mask_path)

        print("\nCaso", i + 1)
        print("patient_id:", row["patient_id"])
        print("pathology:", row["pathology"])
        print("image file path:", full_image_path)
        print("ROI mask file path:", roi_mask_path)
        print("DICOM imagen completa encontrado:", full_dicom)
        print("DICOM máscara encontrado:", mask_dicom)