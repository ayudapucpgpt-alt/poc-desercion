from __future__ import annotations
from fastapi.middleware.cors import CORSMiddleware
import json
from pathlib import Path
from typing import Literal

import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
BOOSTER_PATH = BASE_DIR / "xgb_desercion_proxy_v3_booster.json"
ARTIFACTS_PATH = BASE_DIR / "xgb_desercion_proxy_v3_artifacts.json"


with open(ARTIFACTS_PATH, "r", encoding="utf-8") as f:
    ARTIFACTS = json.load(f)

FEATURE_COLS = ARTIFACTS["feature_cols"]
THRESHOLD = float(ARTIFACTS["threshold"])
MEDIANS = {k: float(v) for k, v in ARTIFACTS["medians"].items()}

MODEL = xgb.Booster()
MODEL.load_model(str(BOOSTER_PATH))


FORM_CONFIG = {
    "anyo_ingreso": {"type": "number", "label": "Año de ingreso", "min": 2013, "max": 2021, "step": 1},
    "tipo_ingreso": {"type": "select", "label": "Tipo de ingreso", "options": ["NAP", "None", "NLE", "NTE", "NRO", "NAI", "NCA", "BMA", "NSC", "NCF", "NSA", "NAD", "ANT", "NUE", "ASA"]},
    "dedicacion": {"type": "select", "label": "Dedicación", "options": ["TC", "TP"]},
    "desplazado_hash": {"type": "select", "label": "Desplazado", "options": ["A", "B"]},
    "curso_mas_bajo": {"type": "number", "label": "Curso más bajo", "min": 1, "max": 4, "step": 1},
    "curso_mas_alto": {"type": "number", "label": "Curso más alto", "min": 1, "max": 5, "step": 1},
    "cred_mat_total": {"type": "number", "label": "Créditos matriculados totales", "min": 9, "max": 79, "step": 1},
    "cred_sup_total": {"type": "number", "label": "Créditos aprobados totales", "min": 0, "max": 66, "step": 1},
    "cred_pend_sup_tit": {"type": "number", "label": "Créditos pendientes para titularse", "min": 6, "max": 259, "step": 1},
    "cred_mat_sem_a": {"type": "number", "label": "Créditos matriculados semestre A", "min": 0, "max": 43, "step": 1},
    "cred_mat_sem_b": {"type": "number", "label": "Créditos matriculados semestre B", "min": 4, "max": 43, "step": 1},
    "cred_sup_sem_a": {"type": "number", "label": "Créditos aprobados semestre A", "min": 0, "max": 39, "step": 1},
    "cred_sup_sem_b": {"type": "number", "label": "Créditos aprobados semestre B", "min": 0, "max": 36, "step": 1},
    "rendimiento_total": {"type": "number", "label": "Rendimiento total", "min": 0, "max": 100, "step": 0.1},
    "rendimiento_cuat_a": {"type": "number", "label": "Rendimiento cuatrimestre A", "min": 0, "max": 100, "step": 0.1},
    "rendimiento_cuat_b": {"type": "number", "label": "Rendimiento cuatrimestre B", "min": 0, "max": 100, "step": 0.1},
    "rend_total_ultimo": {"type": "number", "label": "Rendimiento último periodo", "min": 0, "max": 100, "step": 0.1},
    "rend_total_penultimo": {"type": "number", "label": "Rendimiento penúltimo periodo", "min": 0, "max": 100, "step": 0.1},
    "lms_total_eventos": {"type": "number", "label": "Eventos totales en LMS", "min": 0, "max": 1293, "step": 1},
    "lms_meses_activos": {"type": "number", "label": "Meses activos en LMS", "min": 0, "max": 10, "step": 1},
    "lms_total_visitas": {"type": "number", "label": "Visitas totales en LMS", "min": 0, "max": 258, "step": 1},
    "lms_total_minutos": {"type": "number", "label": "Minutos totales en LMS", "min": 0, "max": 10470, "step": 1},
    "lms_total_entregas": {"type": "number", "label": "Entregas totales en LMS", "min": 0, "max": 27, "step": 1},
    "wifi_dias_totales": {"type": "number", "label": "Días totales con actividad WiFi", "min": 0, "max": 180, "step": 1},
    "wifi_meses_activos": {"type": "number", "label": "Meses activos con WiFi", "min": 0, "max": 10, "step": 1},
}


class PredictionInput(BaseModel):
    anyo_ingreso: int = Field(..., ge=2013, le=2021)
    tipo_ingreso: str
    dedicacion: Literal["TC", "TP"]
    desplazado_hash: Literal["A", "B"]
    curso_mas_bajo: float = Field(..., ge=1, le=4)
    curso_mas_alto: float = Field(..., ge=1, le=5)
    cred_mat_total: float = Field(..., ge=9, le=79)
    cred_sup_total: float = Field(..., ge=0, le=66)
    cred_pend_sup_tit: float = Field(..., ge=6, le=259)
    cred_mat_sem_a: float = Field(..., ge=0, le=43)
    cred_mat_sem_b: float = Field(..., ge=4, le=43)
    cred_sup_sem_a: float = Field(..., ge=0, le=39)
    cred_sup_sem_b: float = Field(..., ge=0, le=36)
    rendimiento_total: float = Field(..., ge=0, le=100)
    rendimiento_cuat_a: float = Field(..., ge=0, le=100)
    rendimiento_cuat_b: float = Field(..., ge=0, le=100)
    rend_total_ultimo: float = Field(..., ge=0, le=100)
    rend_total_penultimo: float = Field(..., ge=0, le=100)
    lms_total_eventos: float = Field(..., ge=0, le=1293)
    lms_meses_activos: float = Field(..., ge=0, le=10)
    lms_total_visitas: float = Field(..., ge=0, le=258)
    lms_total_minutos: float = Field(..., ge=0, le=10470)
    lms_total_entregas: float = Field(..., ge=0, le=27)
    wifi_dias_totales: float = Field(..., ge=0, le=180)
    wifi_meses_activos: float = Field(..., ge=0, le=10)


class PredictionOutput(BaseModel):
    probabilidad_desercion: float
    prediccion_binaria: int
    nivel_riesgo: str
    umbral: float
    factores_principales: list[str]


def get_risk_label(prob: float, threshold: float) -> str:
    if prob < 0.40:
        return "Bajo"
    if prob < threshold:
        return "Medio"
    return "Alto"


def build_form_to_feature_vector(user_input: dict) -> np.ndarray:
    row = {col: MEDIANS.get(col, 0.0) for col in FEATURE_COLS}

    direct_fields = [
        "anyo_ingreso", "curso_mas_bajo", "curso_mas_alto",
        "cred_mat_total", "cred_sup_total", "cred_pend_sup_tit",
        "cred_mat_sem_a", "cred_mat_sem_b", "cred_sup_sem_a", "cred_sup_sem_b",
        "rendimiento_total", "rendimiento_cuat_a", "rendimiento_cuat_b",
        "rend_total_ultimo", "rend_total_penultimo",
        "lms_total_eventos", "lms_meses_activos", "lms_total_visitas",
        "lms_total_minutos", "lms_total_entregas",
        "wifi_dias_totales", "wifi_meses_activos",
    ]

    for col in direct_fields:
        if col in user_input:
            row[col] = float(user_input[col])

    if "ratio_creditos_aprobados" in FEATURE_COLS:
        row["ratio_creditos_aprobados"] = min(max(row["cred_sup_total"] / (row["cred_mat_total"] + 1e-6), 0), 1)

    if "ratio_creditos_pendientes" in FEATURE_COLS:
        row["ratio_creditos_pendientes"] = min(max(row["cred_pend_sup_tit"] / (row["cred_pend_sup_tit"] + row["cred_sup_total"] + 1e-6), 0), 1)

    if "tendencia_rendimiento" in FEATURE_COLS:
        row["tendencia_rendimiento"] = row["rend_total_ultimo"] - row["rend_total_penultimo"]

    if "bajo_rendimiento" in FEATURE_COLS:
        row["bajo_rendimiento"] = 1.0 if row["rendimiento_total"] < 50 else 0.0

    if "lms_ratio_entregas" in FEATURE_COLS:
        row["lms_ratio_entregas"] = min(max(row["lms_total_entregas"] / (row["lms_total_eventos"] + 1e-6), 0), 1)

    tipo_ingreso_map = {
        "NAP": 0, "None": 1, "NLE": 2, "NTE": 3, "NRO": 4,
        "NAI": 5, "NCA": 6, "BMA": 7, "NSC": 8, "NCF": 9,
        "NSA": 10, "NAD": 11, "ANT": 12, "NUE": 13, "ASA": 14
    }
    dedicacion_map = {"TC": 0, "TP": 1}
    desplazado_map = {"A": 0, "B": 1}

    if "tipo_ingreso_enc" in FEATURE_COLS:
        row["tipo_ingreso_enc"] = float(tipo_ingreso_map.get(user_input.get("tipo_ingreso"), 0))
    if "dedicacion_enc" in FEATURE_COLS:
        row["dedicacion_enc"] = float(dedicacion_map.get(user_input.get("dedicacion"), 0))
    if "desplazado_enc" in FEATURE_COLS:
        row["desplazado_enc"] = float(desplazado_map.get(user_input.get("desplazado_hash"), 0))

    vector = np.array([[row[col] for col in FEATURE_COLS]], dtype=np.float32)
    return vector


def explain_prediction(user_input: dict, top_n: int = 3) -> list[str]:
    messages = []

    if user_input.get("rendimiento_total", 100) < 50:
        messages.append("Rendimiento académico general bajo")
    if user_input.get("cred_pend_sup_tit", 0) > 120:
        messages.append("Alta cantidad de créditos pendientes para titularse")
    if user_input.get("lms_meses_activos", 10) <= 3:
        messages.append("Interacción baja con la plataforma virtual")
    if user_input.get("wifi_meses_activos", 10) <= 3:
        messages.append("Baja actividad institucional registrada")
    if user_input.get("cred_sup_total", 999) < 20:
        messages.append("Bajo avance acumulado en créditos aprobados")

    if not messages:
        messages.append("Perfil académico y de actividad dentro de rangos favorables")

    return messages[:top_n]


def predict_from_form(user_input: dict) -> dict:
    vector = build_form_to_feature_vector(user_input)
    dmatrix = xgb.DMatrix(vector, feature_names=FEATURE_COLS)
    prob = float(MODEL.predict(dmatrix)[0])
    pred = int(prob >= THRESHOLD)

    return {
        "probabilidad_desercion": round(prob, 4),
        "prediccion_binaria": pred,
        "nivel_riesgo": get_risk_label(prob, THRESHOLD),
        "umbral": THRESHOLD,
        "factores_principales": explain_prediction(user_input),
    }


app = FastAPI(title="PoC Predicción de Deserción", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://https://v0-student-dropout-prediction-tau.vercel.app/",
        "http://localhost:3000",
    ],
    allow_credentials=true,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def root():
    return {"message": "API activa", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/form-config")
def form_config():
    return FORM_CONFIG


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    return predict_from_form(payload.model_dump())
