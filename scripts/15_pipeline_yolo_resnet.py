from pathlib import Path
import re
import csv
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

# ============================================================
# CONFIGURACION
# ============================================================
BASE = Path(r"C:\Users\Ismael\OneDrive - UNIVERSIDAD DE SEVILLA\Escritorio\MASTER - CLOUD, DATOS Y GESTION\TFM\data\CBIS-DDSM")

YOLO_WEIGHTS = BASE / "mejorado_detect" / "yolo11n_mejorado_ep30-2" / "weights" / "best.pt"
RESNET_WEIGHTS = BASE / "models" / "resnet18_roi_finetuning_cpu.pth"

# Crea esta estructura con tus 50 imágenes y 50 labels:
# CBIS-DDSM/test_sample/images/*.png
# CBIS-DDSM/test_sample/labels/*.txt
IMAGES_DIR = BASE / "test_sample" / "images"
LABELS_DIR = BASE / "test_sample" / "labels"

CALC_CSV = BASE / "csv" / "calc_case_description_test_set.csv"
MASS_CSV = BASE / "csv" / "mass_case_description_test_set.csv"

OUT_DIR = BASE / "experimento_concatenado"

# Tu tutor propuso > 50 %. Además probamos umbrales inferiores para comprobar
# si 0.50 resulta demasiado restrictivo. 0.15 queda cerca del mejor F1 que
# observaste anteriormente (~0.149), y 0.10 ya se estudió en tu TFM.
THRESHOLDS = [0.50, 0.25, 0.15, 0.10]
MIN_YOLO_CONF = min(THRESHOLDS)
IOU_THRESHOLD = 0.50
IMGSZ = 640
DEVICE = "cpu"

# Orden esperado del ImageFolder usado en clasificación: ['benign', 'malignant']
CLASS_NAMES = ["benign", "malignant"]

# ============================================================
# UTILIDADES
# ============================================================
def yolo_label_to_xyxy(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    _, xc, yc, w, h = map(float, parts[:5])
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def load_gt_boxes(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        box = yolo_label_to_xyxy(line, img_w, img_h)
        if box is not None:
            boxes.append(box)
    return boxes


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(preds, gt_boxes, iou_thr=0.50):
    """
    preds: lista de dicts con box/conf/resnet_...
    Devuelve cada predicción como TP o FP y los GT no emparejados como FN.
    Emparejamiento greedy por confianza e IoU.
    """
    used_gt = set()
    matched_preds = []

    for pred in sorted(preds, key=lambda x: x["yolo_conf"], reverse=True):
        best_iou = 0.0
        best_gt = None
        for j, gt in enumerate(gt_boxes):
            if j in used_gt:
                continue
            iou = box_iou(pred["box"], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = j

        out = dict(pred)
        out["iou"] = best_iou
        if best_gt is not None and best_iou >= iou_thr:
            out["det_status"] = "TP"
            out["matched_gt"] = best_gt
            used_gt.add(best_gt)
        else:
            out["det_status"] = "FP"
            out["matched_gt"] = None
        matched_preds.append(out)

    fn_gt = [j for j in range(len(gt_boxes)) if j not in used_gt]
    return matched_preds, fn_gt


def normalize_pathology(value):
    value = str(value).strip().upper()
    if value in {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}:
        return "benign"
    if value == "MALIGNANT":
        return "malignant"
    return "unknown"


def parse_image_identity(stem):
    # Ej.: Calc-Test_P_00038_LEFT_CC
    m = re.match(r"^(Calc|Mass)-Test_(P_\d+)_(LEFT|RIGHT)_(CC|MLO)$", stem)
    if not m:
        return None
    kind, patient_id, laterality, view = m.groups()
    return kind, patient_id, laterality, view


def true_pathology_for_image(stem, calc_df, mass_df):
    ident = parse_image_identity(stem)
    if ident is None:
        return "unknown", 0
    kind, patient_id, laterality, view = ident
    df = calc_df if kind == "Calc" else mass_df

    subset = df[
        (df["patient_id"].astype(str) == patient_id)
        & (df["left or right breast"].astype(str).str.upper() == laterality)
        & (df["image view"].astype(str).str.upper() == view)
    ]
    if subset.empty:
        return "unknown", 0

    labels = sorted(set(normalize_pathology(v) for v in subset["pathology"].tolist()))
    labels = [x for x in labels if x != "unknown"]
    if len(labels) == 1:
        return labels[0], len(subset)
    if len(labels) > 1:
        # Si alguna imagen tuviera lesiones con patologías distintas, no sería correcto
        # asignar una etiqueta única a toda la mamografía sin reconstruir la asociación
        # exacta entre abnormality id y bounding box.
        return "mixed", len(subset)
    return "unknown", len(subset)


def build_resnet(weights_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def classify_crop(model, pil_crop):
    # En entrenamiento las ROI se convertían a 3 canales para ResNet18.
    crop = pil_crop.convert("RGB")
    x = resnet_transform(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    return CLASS_NAMES[idx], float(probs[idx].item()), float(probs[1].item())


def safe_crop(img, box):
    x1, y1, x2, y2 = box
    x1 = max(0, min(img.width - 1, int(round(x1))))
    y1 = max(0, min(img.height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(img.width, int(round(x2))))
    y2 = max(y1 + 1, min(img.height, int(round(y2))))
    return img.crop((x1, y1, x2, y2))


def draw_visual(img, gt_boxes, matched_preds, fn_gt, title_text=None):
    vis = img.convert("RGB").copy()
    d = ImageDraw.Draw(vis)

    # Ground truth: verde
    for i, b in enumerate(gt_boxes):
        d.rectangle(b, outline="green", width=4)
        d.text((b[0] + 3, max(0, b[1] - 16)), f"GT {i+1}", fill="green")

    # Predicciones: azul si TP, rojo si FP
    for p in matched_preds:
        b = p["box"]
        color = "blue" if p["det_status"] == "TP" else "red"
        label = (
            f"{p['det_status']} Y:{p['yolo_conf']:.2f} "
            f"R:{p['resnet_label']}({p['resnet_conf']:.2f})"
        )
        d.rectangle(b, outline=color, width=4)
        d.text((b[0] + 3, min(vis.height - 18, b[1] + 3)), label, fill=color)

    # Marcar explícitamente GT no recuperados
    for j in fn_gt:
        b = gt_boxes[j]
        d.text((b[0] + 3, min(vis.height - 18, b[3] + 3)), "FN", fill="orange")

    if title_text:
        d.text((10, 10), title_text, fill="yellow")
    return vis


# ============================================================
# EXPERIMENTO
# ============================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"No se encuentra YOLO: {YOLO_WEIGHTS}")
    if not RESNET_WEIGHTS.exists():
        raise FileNotFoundError(f"No se encuentra ResNet18: {RESNET_WEIGHTS}")
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"No se encuentra la carpeta de imágenes: {IMAGES_DIR}")
    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"No se encuentra la carpeta de labels: {LABELS_DIR}")

    calc_df = pd.read_csv(CALC_CSV)
    mass_df = pd.read_csv(MASS_CSV)

    yolo = YOLO(str(YOLO_WEIGHTS))
    resnet = build_resnet(RESNET_WEIGHTS)

    # Acumuladores por umbral
    rows_by_thr = {thr: [] for thr in THRESHOLDS}
    totals = {thr: defaultdict(int) for thr in THRESHOLDS}

    # Ejecutamos YOLO una sola vez al umbral mínimo y luego filtramos.
    results = yolo.predict(
        source=str(IMAGES_DIR),
        conf=MIN_YOLO_CONF,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
        stream=True,
    )

    for result in results:
        img_path = Path(result.path)
        stem = img_path.stem
        label_path = LABELS_DIR / f"{stem}.txt"
        img = Image.open(img_path).convert("L")
        gt_boxes = load_gt_boxes(label_path, img.width, img.height)
        true_pathology, n_csv_rows = true_pathology_for_image(stem, calc_df, mass_df)

        # Todas las predicciones desde conf >= 0.10, clasificadas una única vez con ResNet18.
        all_preds = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for det_idx, (box, yconf) in enumerate(zip(xyxy, confs), start=1):
                box = [float(v) for v in box.tolist()]
                crop = safe_crop(img, box)
                rlabel, rconf, p_malignant = classify_crop(resnet, crop)
                all_preds.append({
                    "det_idx": det_idx,
                    "box": box,
                    "yolo_conf": float(yconf),
                    "resnet_label": rlabel,
                    "resnet_conf": rconf,
                    "resnet_p_malignant": p_malignant,
                })

        for thr in THRESHOLDS:
            thr_tag = f"conf_{int(round(thr*100)):02d}"
            thr_dir = OUT_DIR / thr_tag
            vis_dir = thr_dir / "visualizaciones"
            crop_dir = thr_dir / "crops"
            vis_dir.mkdir(parents=True, exist_ok=True)
            crop_dir.mkdir(parents=True, exist_ok=True)

            preds = [p for p in all_preds if p["yolo_conf"] >= thr]
            matched_preds, fn_gt = match_predictions(preds, gt_boxes, IOU_THRESHOLD)

            totals[thr]["images"] += 1
            totals[thr]["gt"] += len(gt_boxes)
            totals[thr]["pred"] += len(matched_preds)
            totals[thr]["tp"] += sum(p["det_status"] == "TP" for p in matched_preds)
            totals[thr]["fp"] += sum(p["det_status"] == "FP" for p in matched_preds)
            totals[thr]["fn"] += len(fn_gt)
            if matched_preds:
                totals[thr]["images_with_pred"] += 1

            # Guardar una fila por predicción
            for p in matched_preds:
                cls_correct = ""
                if p["det_status"] == "TP" and true_pathology in {"benign", "malignant"}:
                    cls_correct = int(p["resnet_label"] == true_pathology)
                    totals[thr]["tp_classified"] += 1
                    totals[thr]["tp_cls_correct"] += cls_correct
                    if true_pathology == "malignant":
                        totals[thr]["tp_malignant"] += 1
                        if p["resnet_label"] == "malignant":
                            totals[thr]["tp_malignant_correct"] += 1

                rows_by_thr[thr].append({
                    "image": img_path.name,
                    "true_pathology": true_pathology,
                    "csv_rows_for_view": n_csv_rows,
                    "gt_boxes_in_image": len(gt_boxes),
                    "prediction_index": p["det_idx"],
                    "yolo_conf": round(p["yolo_conf"], 6),
                    "det_status": p["det_status"],
                    "iou_best_gt": round(p["iou"], 6),
                    "resnet_prediction": p["resnet_label"],
                    "resnet_confidence": round(p["resnet_conf"], 6),
                    "resnet_p_malignant": round(p["resnet_p_malignant"], 6),
                    "resnet_correct_if_tp": cls_correct,
                })

                # Guardar crop para poder inspeccionarlo después
                crop = safe_crop(img, p["box"]).convert("RGB")
                crop_name = (
                    f"{stem}__det{p['det_idx']}__{p['det_status']}__"
                    f"y{p['yolo_conf']:.2f}__{p['resnet_label']}_{p['resnet_conf']:.2f}.png"
                )
                crop.save(crop_dir / crop_name)

            # Si no hay predicciones, guardar también una fila de imagen para dejar constancia
            if not matched_preds:
                rows_by_thr[thr].append({
                    "image": img_path.name,
                    "true_pathology": true_pathology,
                    "csv_rows_for_view": n_csv_rows,
                    "gt_boxes_in_image": len(gt_boxes),
                    "prediction_index": "",
                    "yolo_conf": "",
                    "det_status": "NO_PRED",
                    "iou_best_gt": "",
                    "resnet_prediction": "",
                    "resnet_confidence": "",
                    "resnet_p_malignant": "",
                    "resnet_correct_if_tp": "",
                })

            vis = draw_visual(
                img,
                gt_boxes,
                matched_preds,
                fn_gt,
                title_text=f"conf >= {thr:.2f} | GT={len(gt_boxes)} pred={len(matched_preds)}",
            )
            vis.save(vis_dir / f"{stem}.png")

    # ========================================================
    # SALIDAS
    # ========================================================
    summary_rows = []
    for thr in THRESHOLDS:
        t = totals[thr]
        precision = t["tp"] / (t["tp"] + t["fp"]) if (t["tp"] + t["fp"]) else 0.0
        recall = t["tp"] / (t["tp"] + t["fn"]) if (t["tp"] + t["fn"]) else 0.0
        cls_acc = t["tp_cls_correct"] / t["tp_classified"] if t["tp_classified"] else math.nan
        mal_recall = (
            t["tp_malignant_correct"] / t["tp_malignant"]
            if t["tp_malignant"] else math.nan
        )

        summary_rows.append({
            "threshold": thr,
            "images": t["images"],
            "images_with_prediction": t["images_with_pred"],
            "gt_lesions": t["gt"],
            "predicted_boxes": t["pred"],
            "TP_detection": t["tp"],
            "FP_detection": t["fp"],
            "FN_detection": t["fn"],
            "detection_precision": round(precision, 4),
            "detection_recall": round(recall, 4),
            "TP_sent_to_resnet": t["tp_classified"],
            "resnet_correct_on_TP": t["tp_cls_correct"],
            "resnet_accuracy_on_detected_TP": "" if math.isnan(cls_acc) else round(cls_acc, 4),
            "malignant_TP_sent_to_resnet": t["tp_malignant"],
            "malignant_correct_by_resnet": t["tp_malignant_correct"],
            "resnet_recall_malignant_on_detected_TP": "" if math.isnan(mal_recall) else round(mal_recall, 4),
        })

        detail_csv = OUT_DIR / f"detalle_pipeline_conf_{int(round(thr*100)):02d}.csv"
        pd.DataFrame(rows_by_thr[thr]).to_csv(detail_csv, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "resumen_pipeline_yolo_resnet.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n================ RESULTADOS ================")
    print(summary_df.to_string(index=False))
    print("\nResultados guardados en:")
    print(OUT_DIR)
    print("\nArchivo principal:")
    print(summary_path)
    print("\nIMPORTANTE:")
    print("- TP/FP/FN de detección usan IoU >= 0.50.")
    print("- La exactitud de ResNet se calcula SOLO sobre cajas YOLO que son TP de detección.")
    print("- Un falso positivo de YOLO también recibe benign/malignant de ResNet, pero esa etiqueta NO puede considerarse correcta/incorrecta porque no corresponde a una lesión real.")
    print("- Si la columna true_pathology aparece como 'mixed', no uses esa imagen para calcular exactitud de clasificación sin reconstruir la correspondencia lesión-abnormality id.")


if __name__ == "__main__":
    main()
