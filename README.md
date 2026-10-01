# Clasificación y detección de lesiones mamarias mediante Deep Learning

Repositorio asociado al Trabajo Fin de Máster centrado en la aplicación de técnicas de Deep Learning al análisis de mamografías del conjunto de datos **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**.

El proyecto aborda principalmente dos tareas:

- **Clasificación de regiones de interés (ROI)** para distinguir entre lesiones benignas y malignas.
- **Detección y localización de anomalías mamarias** mediante modelos de detección de objetos.

Para ello se han utilizado técnicas de aprendizaje profundo, transferencia de aprendizaje (*transfer learning*) y ajuste fino (*fine-tuning*) de redes neuronales convolucionales.

---

## Objetivos

Los principales objetivos del proyecto son:

1. Analizar y explorar el conjunto de datos CBIS-DDSM.
2. Preparar las regiones de interés correspondientes a las anomalías mamarias.
3. Entrenar modelos de clasificación basados en redes neuronales convolucionales.
4. Evaluar el rendimiento de los modelos mediante distintas métricas.
5. Aplicar técnicas de *fine-tuning* para mejorar los resultados.
6. Preparar el conjunto de datos para detección de lesiones.
7. Entrenar modelos de detección basados en YOLO.
8. Integrar clasificación y detección dentro de un pipeline completo.

---

## Estructura del repositorio

```text
.
├── notebooks/
│   └── 01_exploracion_dataset.ipynb
│
├── scripts/
│   ├── 02_preparar_roi_clasificacion.py
│   ├── 02-02_comprobar_roi.py
│   ├── 03_comprobar_dataset_roi.py
│   ├── 04_entrenar_resnet50_roi.py
│   ├── 05_evaluar_modelo_roi.py
│   ├── 06_entrenar_resnet18_mejorado.py
│   ├── 07_evaluar_modelo_mejorado.py
│   ├── 08_graficas_entrenamiento.py
│   ├── 09_entrenar_resnet18_finetuning.py
│   ├── 10_evaluar_modelo_finetuning.py
│   ├── 11_explorar_deteccion.py
│   ├── 12_preparar_dataset_deteccion.py
│   ├── 12_preparar_dataset_deteccion_completo.py
│   ├── 13_comprobar_dataset_deteccion.py
│   ├── 13_comprobar_dataset_deteccion_completo.py
│   └── 15_pipeline_yolo_resnet.py
│
├── 14_dataset_yolo.yaml
├── requirements.txt
├── .gitignore
└── README.md