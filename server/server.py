# pylint: disable=no-member,no-name-in-module
"""
HTTP server that bridges Unity (drone simulation) with the YOLO fire model.

Unity sends a frame captured by the drone camera (multipart/form-data or
raw bytes / base64 JSON) and the server returns the most critical fire
point detected, so the drone can autonomously navigate towards it.

Endpoints
---------
GET  /            -> health check
GET  /health      -> health check with model status
POST /detect      -> multipart upload (field: "image"), recommended for Unity
POST /detect/b64  -> JSON body { "image": "<base64>" }, alternative

Response (JSON)
---------------
{
    "fire_detected": true,
    "image_width":  1280,
    "image_height": 720,
    "detections": [
        {
            "cx": 640, "cy": 360,
            "cx_norm": 0.5, "cy_norm": 0.5,
            "area": 12345, "radius": 62,
            "confidence": 0.87,
            "bbox": [x1, y1, x2, y2]
        }
    ],
    "primary_target": { ... same shape as a detection ... } | null,
    "inference_ms": 18.4
}
"""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO


# ── Configuración ──────────────────────────────────────────────
WEIGHTS_PATH   = Path(__file__).resolve().parent.parent / "weights" / "best.pt"
FIRE_HSV_LOW   = np.array([0,  120, 120])
FIRE_HSV_HIGH  = np.array([35, 255, 255])
MIN_FIRE_AREA  = 500          # px² mínimo para considerar detección válida
CONF_THRESHOLD = 0.35         # confianza mínima YOLO


# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="Fire Detection Server",
    description="YOLO fire detection bridge for Unity drone simulation",
    version="1.0.0",
)

# Unity normalmente corre en otro host/puerto → habilitar CORS abierto
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga única del modelo al levantar el server
print(f"[server] Loading YOLO model from: {WEIGHTS_PATH}")
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo en {WEIGHTS_PATH}. "
        f"Asegúrate de tener el archivo best.pt en la carpeta weights/."
    )
model = YOLO(str(WEIGHTS_PATH))
print("[server] Model loaded successfully")


# ── Lógica de detección ────────────────────────────────────────
def get_critical_point(mask: np.ndarray, offset_x: int, offset_y: int):
    """Centroide ponderado por intensidad → núcleo más caliente del fuego."""
    area = int(cv2.countNonZero(mask))
    if area < MIN_FIRE_AREA:
        return None

    coords  = np.column_stack(np.where(mask > 0))     # [row, col]
    weights = mask[mask > 0].astype(np.float64)
    total   = weights.sum()
    if total == 0:
        return None

    cy_local = (coords[:, 0] * weights).sum() / total
    cx_local = (coords[:, 1] * weights).sum() / total

    cx = int(offset_x + cx_local)
    cy = int(offset_y + cy_local)
    radius = int(np.sqrt(area / np.pi))
    return cx, cy, area, radius


def analyze_frame(frame: np.ndarray) -> dict:
    """Corre YOLO + análisis HSV. Devuelve dict listo para JSON."""
    t0 = time.perf_counter()
    h, w = frame.shape[:2]

    results = model(frame, verbose=False, conf=CONF_THRESHOLD)
    detections: list[dict] = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, FIRE_HSV_LOW, FIRE_HSV_HIGH)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        critical = get_critical_point(mask, offset_x=x1, offset_y=y1)
        if critical is None:
            # No hay máscara HSV suficiente → usar el centro del bbox como fallback
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            area   = (x2 - x1) * (y2 - y1)
            radius = int(np.sqrt(area / np.pi))
        else:
            cx, cy, area, radius = critical

        detections.append({
            "cx": cx,
            "cy": cy,
            "cx_norm": round(cx / w, 4),
            "cy_norm": round(cy / h, 4),
            "area": int(area),
            "radius": int(radius),
            "confidence": float(box.conf[0]),
            "bbox": [x1, y1, x2, y2],
        })

    # El blanco principal = la detección con mayor área (más peligrosa / cercana)
    primary = max(detections, key=lambda d: d["area"], default=None)

    return {
        "fire_detected": len(detections) > 0,
        "image_width":   w,
        "image_height":  h,
        "detections":    detections,
        "primary_target": primary,
        "inference_ms":  round((time.perf_counter() - t0) * 1000, 2),
    }


# ── Helpers para decodificar la imagen entrante ────────────────
def decode_image_bytes(data: bytes) -> np.ndarray:
    arr   = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")
    return frame


class Base64Image(BaseModel):
    image: str  # base64 (con o sin prefijo data:image/...;base64,)


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Fire Detection Server is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    """
    Endpoint recomendado para Unity:
    POST multipart/form-data con un campo 'image' que contenga el JPG/PNG.
    """
    data  = await image.read()
    frame = decode_image_bytes(data)
    return analyze_frame(frame)


@app.post("/detect/b64")
def detect_b64(payload: Base64Image):
    """
    Alternativa por si Unity envía la imagen como base64 dentro de un JSON.
    """
    b64 = payload.image
    if "," in b64:                     # quita prefijo data:image/...;base64,
        b64 = b64.split(",", 1)[1]
    try:
        data = base64.b64decode(b64)
    except Exception as e:             # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=f"Base64 inválido: {e}")
    frame = decode_image_bytes(data)
    return analyze_frame(frame)