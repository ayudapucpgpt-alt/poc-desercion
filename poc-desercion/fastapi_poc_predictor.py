from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgb_desercion_proxy_v3_package.pkl"

# -----------------------------------------------------------------------------
# Cargar paquete del modelo
# Se espera un pickle con al menos:
# - model
# - feature_cols
# - threshold
# - medians
# -----------------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    package = pickle.load(f)

model = package["model"]
feature_cols = package["feature_cols"]
threshold = float(package["threshold"])
medians = package["medians"]

# Asegurar formato dict para imputación
if hasattr(medians, "to_dict"):
    medians = medians.to_dict()
else:
    medians = dict(medians)

# -----------------------------------------------------------------------------
# Metadata para formulario / frontend
# -----------------------------------------------------------------------------
FORM_CONFIG = {
    "anyo_ingreso": {
        "type": "number",
        "label": "Año de ingreso",
        "min": 2013,
        "max": 2021,
        "step": 1,
    },
    "tipo_ingreso": {
        "type": "select",
        "label": "Tipo de ingreso",
        "options": ["NAP", "None", "NLE", "NTE", "NRO", "NAI", "NCA", "BMA", "NSC", "NCF", "NSA", "NAD", "ANT", "NUE", "ASA"],
    },
    "dedicacion": {
        "type": "select",
        "label": "Dedicación",
        "options": ["TC", "TP"],
    },
    "desplazado_hash": {
        "type": "select",
        "label": "Desplazado",
        "options": ["A", "B"],
    },
    "curso_mas_bajo": {
        "type": "number",
        "label": "Curso más bajo",
        "min": 1,
        "max": 4,
        "step": 1,
    },
    "curso_mas_alto": {
        "type": "number",
        "label": "Curso más alto",
        "min": 1,
        "max": 5,
        "step": 1,
    },
    "cred_mat_total": {
        "type": "number",
        "label": "Créditos matriculados totales",
        "min": 9,
        "max": 79,
        "step": 1,
    },
    "cred_sup_total": {
        "type": "number",
        "label": "Créditos aprobados totales",
        "min": 0,
        "max": 66,
        "step": 1,
    },
    "cred_pend_sup_tit": {
        "type": "number",
        "label": "Créditos pendientes para titularse",
        "min": 6,
        "max": 259,
        "step": 1,
    },
    "cred_mat_sem_a": {
        "type": "number",
        "label": "Créditos matriculados semestre A",
        "min": 0,
        "max": 43,
        "step": 1,
    },
    "cred_mat_sem_b": {
        "type": "number",
        "label": "Créditos matriculados semestre B",
        "min": 4,
        "max": 43,
        "step": 1,
    },
    "cred_sup_sem_a": {
        "type": "number",
        "label": "Créditos aprobados semestre A",
        "min": 0,
        "max": 39,
        "step": 1,
    },
    "cred_sup_sem_b": {
        "type": "number",
        "label": "Créditos aprobados semestre B",
        "min": 0,
        "max": 36,
        "step": 1,
    },
    "rendimiento_total": {
        "type": "number",
        "label": "Rendimiento total",
        "min": 0,
        "max": 100,
        "step": 0.1,
    },
    "rendimiento_cuat_a": {
        "type": "number",
        "label": "Rendimiento cuatrimestre A",
        "min": 0,
        "max": 100,
        "step": 0.1,
    },
    "rendimiento_cuat_b": {
        "type": "number",
        "label": "Rendimiento cuatrimestre B",
        "min": 0,
        "max": 100,
        "step": 0.1,
    },
    "rend_total_ultimo": {
        "type": "number",
        "label": "Rendimiento último periodo",
        "min": 0,
        "max": 100,
        "step": 0.1,
    },
    "rend_total_penultimo": {
        "type": "number",
        "label": "Rendimiento penúltimo periodo",
        "min": 0,
        "max": 100,
        "step": 0.1,
    },
    "lms_total_eventos": {
        "type": "number",
        "label": "Eventos totales en LMS",
        "min": 0,
        "max": 1293,
        "step": 1,
    },
    "lms_meses_activos": {
        "type": "number",
        "label": "Meses activos en LMS",
        "min": 0,
        "max": 10,
        "step": 1,
    },
    "lms_total_visitas": {
        "type": "number",
        "label": "Visitas totales en LMS",
        "min": 0,
        "max": 258,
        "step": 1,
    },
    "lms_total_minutos": {
        "type": "number",
        "label": "Minutos totales en LMS",
        "min": 0,
        "max": 10470,
        "step": 1,
    },
    "lms_total_entregas": {
        "type": "number",
        "label": "Entregas totales en LMS",
        "min": 0,
        "max": 27,
        "step": 1,
    },
    "wifi_dias_totales": {
        "type": "number",
        "label": "Días totales con actividad WiFi",
        "min": 0,
        "max": 180,
        "step": 1,
    },
    "wifi_meses_activos": {
        "type": "number",
        "label": "Meses activos con WiFi",
        "min": 0,
        "max": 10,
        "step": 1,
    },
}

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class PredictionInput(BaseModel):
    anyo_ingreso: int = Field(ge=2013, le=2021)
    tipo_ingreso: Literal["NAP", "None", "NLE", "NTE", "NRO", "NAI", "NCA", "BMA", "NSC", "NCF", "NSA", "NAD", "ANT", "NUE", "ASA"]
    dedicacion: Literal["TC", "TP"]
    desplazado_hash: Literal["A", "B"]
    curso_mas_bajo: int = Field(ge=1, le=4)
    curso_mas_alto: int = Field(ge=1, le=5)
    cred_mat_total: float = Field(ge=9, le=79)
    cred_sup_total: float = Field(ge=0, le=66)
    cred_pend_sup_tit: float = Field(ge=6, le=259)
    cred_mat_sem_a: float = Field(ge=0, le=43)
    cred_mat_sem_b: float = Field(ge=4, le=43)
    cred_sup_sem_a: float = Field(ge=0, le=39)
    cred_sup_sem_b: float = Field(ge=0, le=36)
    rendimiento_total: float = Field(ge=0, le=100)
    rendimiento_cuat_a: float = Field(ge=0, le=100)
    rendimiento_cuat_b: float = Field(ge=0, le=100)
    rend_total_ultimo: float = Field(ge=0, le=100)
    rend_total_penultimo: float = Field(ge=0, le=100)
    lms_total_eventos: float = Field(ge=0, le=1293)
    lms_meses_activos: int = Field(ge=0, le=10)
    lms_total_visitas: float = Field(ge=0, le=258)
    lms_total_minutos: float = Field(ge=0, le=10470)
    lms_total_entregas: float = Field(ge=0, le=27)
    wifi_dias_totales: float = Field(ge=0, le=180)
    wifi_meses_activos: int = Field(ge=0, le=10)


class PredictionOutput(BaseModel):
    probabilidad_desercion: float
    prediccion_binaria: int
    nivel_riesgo: str
    umbral: float
    factores_principales: list[str]


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def build_form_to_model_input(user_input: dict, feature_cols: list[str], medians: dict) -> pd.DataFrame:
    """Convierte inputs amigables en una fila lista para el modelo."""
    row = {col: medians.get(col, 0) for col in feature_cols}

    direct_fields = [
        "anyo_ingreso",
        "curso_mas_bajo",
        "curso_mas_alto",
        "cred_mat_total",
        "cred_sup_total",
        "cred_pend_sup_tit",
        "cred_mat_sem_a",
        "cred_mat_sem_b",
        "cred_sup_sem_a",
        "cred_sup_sem_b",
        "rendimiento_total",
        "rendimiento_cuat_a",
        "rendimiento_cuat_b",
        "rend_total_ultimo",
        "rend_total_penultimo",
        "lms_total_eventos",
        "lms_meses_activos",
        "lms_total_visitas",
        "lms_total_minutos",
        "lms_total_entregas",
        "wifi_dias_totales",
        "wifi_meses_activos",
    ]

    for col in direct_fields:
        if col in feature_cols and col in user_input:
            row[col] = user_input[col]

    # Features derivadas
    if "ratio_creditos_aprobados" in feature_cols:
        row["ratio_creditos_aprobados"] = min(
            max(row["cred_sup_total"] / (row["cred_mat_total"] + 1e-6), 0), 1
        )

    if "ratio_creditos_pendientes" in feature_cols:
        row["ratio_creditos_pendientes"] = min(
            max(row["cred_pend_sup_tit"] / (row["cred_pend_sup_tit"] + row["cred_sup_total"] + 1e-6), 0), 1
        )

    if "tendencia_rendimiento" in feature_cols:
        row["tendencia_rendimiento"] = row["rend_total_ultimo"] - row["rend_total_penultimo"]

    if "bajo_rendimiento" in feature_cols:
        row["bajo_rendimiento"] = int(row["rendimiento_total"] < 50)

    if "lms_ratio_entregas" in feature_cols:
        row["lms_ratio_entregas"] = min(
            max(row["lms_total_entregas"] / (row["lms_total_eventos"] + 1e-6), 0), 1
        )

    # Encodings manuales
    tipo_ingreso_map = {
        "NAP": 0,
        "None": 1,
        "NLE": 2,
        "NTE": 3,
        "NRO": 4,
        "NAI": 5,
        "NCA": 6,
        "BMA": 7,
        "NSC": 8,
        "NCF": 9,
        "NSA": 10,
        "NAD": 11,
        "ANT": 12,
        "NUE": 13,
        "ASA": 14,
    }
    dedicacion_map = {"TC": 0, "TP": 1}
    desplazado_map = {"A": 0, "B": 1}

    if "tipo_ingreso_enc" in feature_cols:
        row["tipo_ingreso_enc"] = tipo_ingreso_map.get(user_input.get("tipo_ingreso"), 0)

    if "dedicacion_enc" in feature_cols:
        row["dedicacion_enc"] = dedicacion_map.get(user_input.get("dedicacion"), 0)

    if "desplazado_enc" in feature_cols:
        row["desplazado_enc"] = desplazado_map.get(user_input.get("desplazado_hash"), 0)

    return pd.DataFrame([[row[col] for col in feature_cols]], columns=feature_cols)


def get_risk_label(prob: float, thr: float) -> str:
    if prob < 0.40:
        return "Bajo"
    if prob < thr:
        return "Medio"
    return "Alto"


def explain_prediction(x_row: pd.DataFrame, top_n: int = 5) -> list[str]:
    """
    Explicación simple para PoC.
    Usa reglas heurísticas legibles para mostrar factores principales.
    """
    r = x_row.iloc[0]
    messages: list[str] = []

    if r.get("cred_pend_sup_tit", 0) > 120:
        messages.append("Muchos créditos pendientes para titularse")
    if r.get("rendimiento_total", 100) < 60:
        messages.append("Rendimiento académico general bajo")
    if r.get("tendencia_rendimiento", 0) < -10:
        messages.append("Caída reciente en el rendimiento")
    if r.get("lms_meses_activos", 10) <= 3:
        messages.append("Poca constancia de actividad en LMS")
    if r.get("lms_total_eventos", 9999) < 100:
        messages.append("Interacción baja con la plataforma virtual")
    if r.get("wifi_meses_activos", 10) <= 3:
        messages.append("Baja actividad institucional registrada")
    if r.get("ratio_creditos_aprobados", 1) < 0.5:
        messages.append("Baja proporción de créditos aprobados")
    if r.get("ratio_creditos_pendientes", 0) > 0.6:
        messages.append("Alta carga pendiente respecto al avance logrado")

    if not messages:
        messages.append("Perfil académico y de actividad dentro de rangos favorables")

    return messages[:top_n]


def predict_from_form(user_input: dict) -> dict:
    x_new = build_form_to_model_input(
        user_input=user_input,
        feature_cols=feature_cols,
        medians=medians,
    )

    prob = float(model.predict_proba(x_new)[:, 1][0])
    pred = int(prob >= threshold)
    risk_label = get_risk_label(prob, threshold)
    factores = explain_prediction(x_new)

    return {
        "probabilidad_desercion": round(prob, 4),
        "prediccion_binaria": pred,
        "nivel_riesgo": risk_label,
        "umbral": threshold,
        "factores_principales": factores,
    }


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
app = FastAPI(
    title="PoC Predicción de Deserción",
    version="0.1.0",
    description="Prueba de concepto para estimar riesgo de deserción estudiantil.",
)


@app.get("/")
def root():
    return {
        "message": "API de predicción de deserción activa",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/form-config")
def form_config():
    return FORM_CONFIG


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    return predict_from_form(payload.model_dump())
