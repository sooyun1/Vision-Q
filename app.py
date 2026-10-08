"""Vision Q: project-asset-backed inspection dashboard."""

import base64
import importlib.util
import io
import json
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).parent
BACKGROUND_IMAGE_PATH = PROJECT_ROOT / "images" / "Background.png"
PROCESS_REVIEW_LOGIC_PATH = PROJECT_ROOT / "pages" / "1_Process_Review.py"
DB_PATH = PROJECT_ROOT / "inspection.db"
MODEL_OPTIONS = {
    "YOLOv5": "best.pt/yolov5n_best.pt",
    "YOLOv8": "best.pt/yolov8n_best.pt",
    "YOLOv10": "best.pt/yolov10_best.pt",
    "YOLOv11": "best.pt/yolo11n_best.pt",
    "YOLOv12": "best.pt/yolov12_best.pt",
    "YOLOv26": "best.pt/yolo26n_best.pt",
}
MODEL_IDENTIFIERS = tuple(Path(model_path).stem for model_path in MODEL_OPTIONS.values())
SAMPLE_IMAGE_DIR = PROJECT_ROOT / "dataset_det" / "test" / "images"
BAD_DATASET_DIR = PROJECT_ROOT / "dataset" / "bad"
GOOD_DATASET_DIR = PROJECT_ROOT / "dataset" / "good"
MODEL_ANALYSES = {
    "yolov5": {
        "display_name": "YOLOv5",
        "results_chart": PROJECT_ROOT / "runs_det" / "v5_experiment" / "results.png",
        "confusion_matrix": PROJECT_ROOT / "runs_det" / "v5_experiment" / "confusion_matrix.png",
        "normalized_confusion_matrix": PROJECT_ROOT / "runs_det" / "v5_experiment" / "confusion_matrix_normalized.png",
        "metrics_csv": PROJECT_ROOT / "runs_det" / "v5_experiment" / "results.csv",
        "performance_curves": ["BoxF1_curve.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png"],
    },
    "yolov8": {
        "display_name": "YOLOv8",
        "results_chart": PROJECT_ROOT / "runs_det" / "v8_experiment" / "results.png",
        "confusion_matrix": PROJECT_ROOT / "runs_det" / "v8_experiment" / "confusion_matrix.png",
        "normalized_confusion_matrix": PROJECT_ROOT / "runs_det" / "v8_experiment" / "confusion_matrix_normalized.png",
        "metrics_csv": PROJECT_ROOT / "runs_det" / "v8_experiment" / "results.csv",
        "performance_curves": ["BoxF1_curve.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png"],
    },
    "yolov10": {
        "display_name": "YOLOv10",
        "results_chart": PROJECT_ROOT / "runs_det" / "v10_experiment" / "results.png",
        "confusion_matrix": PROJECT_ROOT / "runs_det" / "v10_experiment" / "confusion_matrix.png",
        "normalized_confusion_matrix": PROJECT_ROOT / "runs_det" / "v10_experiment" / "confusion_matrix_normalized.png",
        "metrics_csv": PROJECT_ROOT / "runs_det" / "v10_experiment" / "results.csv",
        "performance_curves": ["BoxF1_curve.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png"],
    },
    "yolov11": {
        "display_name": "YOLOv11",
        "results_chart": PROJECT_ROOT / "runs_det" / "v11_experiment" / "results.png",
        "confusion_matrix": PROJECT_ROOT / "runs_det" / "v11_experiment" / "confusion_matrix.png",
        "normalized_confusion_matrix": PROJECT_ROOT / "runs_det" / "v11_experiment" / "confusion_matrix_normalized.png",
        "metrics_csv": PROJECT_ROOT / "runs_det" / "v11_experiment" / "results.csv",
        "performance_curves": ["BoxF1_curve.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png"],
    },
    "yolov12": {
        "display_name": "YOLOv12",
        "results_chart": PROJECT_ROOT / "runs_det" / "v12_experiment" / "results.png",
        "confusion_matrix": PROJECT_ROOT / "runs_det" / "v12_experiment" / "confusion_matrix.png",
        "normalized_confusion_matrix": PROJECT_ROOT / "runs_det" / "v12_experiment" / "confusion_matrix_normalized.png",
        "metrics_csv": PROJECT_ROOT / "runs_det" / "v12_experiment" / "results.csv",
        "performance_curves": ["BoxF1_curve.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png"],
    },
    "yolov26": {
        "display_name": "YOLOv26",
        "results_chart": PROJECT_ROOT / "runs_det" / "v26_experiment" / "results.png",
        "confusion_matrix": PROJECT_ROOT / "runs_det" / "v26_experiment" / "confusion_matrix.png",
        "normalized_confusion_matrix": PROJECT_ROOT / "runs_det" / "v26_experiment" / "confusion_matrix_normalized.png",
        "metrics_csv": PROJECT_ROOT / "runs_det" / "v26_experiment" / "results.csv",
        "performance_curves": ["BoxF1_curve.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png"],
    },
}
RESULT_NORMAL = "정상"
RESULT_DEFECTIVE = "불량"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
COLORS = {
    "background": "#0b1020",
    "surface": "#161d31",
    "surface_alt": "#1d2740",
    "text": "#edf2ff",
    "muted": "#9aa8c7",
    "accent": "#7c5cff",
    "normal": "#34d399",
    "defective": "#fb7185",
}


def apply_custom_style() -> None:
    """Apply the complete typography, component, and navigation design system."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');

        :root {{
            --app-bg: {COLORS['background']};
            --panel: {COLORS['surface']};
            --panel-raised: {COLORS['surface_alt']};
            --text: {COLORS['text']};
            --muted: {COLORS['muted']};
            --accent: {COLORS['accent']};
            --soft-border: rgba(255, 255, 255, 0.10);
            --soft-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }}
        html, body, [class*="css"], .stApp {{
            font-family: 'Inter', 'Noto Sans KR', sans-serif;
            color: var(--text);
        }}
        .stApp {{ background: radial-gradient(circle at 8% 0%, #1d2850 0%, var(--app-bg) 38%); }}
        [data-testid="stHeader"] {{ background: rgba(11, 16, 32, 0.82); backdrop-filter: blur(12px); }}
        .block-container {{ max-width: 1480px; padding: 2.35rem 2.7rem 3.5rem; }}
        .element-container {{ margin-bottom: 0.7rem; }}
        h1 {{
            color: var(--text) !important; font-size: 28px !important; font-weight: 700 !important;
            line-height: 1.25 !important; letter-spacing: -0.035em; margin: 0 0 1.65rem !important;
        }}
        h2, h3 {{
            color: var(--text) !important; font-size: 20px !important; font-weight: 600 !important;
            line-height: 1.4 !important; letter-spacing: -0.02em; margin: 1.6rem 0 0.8rem !important;
        }}
        p, li, [data-testid="stCaptionContainer"] {{ font-size: 14px; line-height: 1.6; }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #151d34 0%, #0b1020 100%);
            border-right: 1px solid var(--soft-border);
            padding: 0.85rem 0.7rem;
        }}
        [data-testid="stSidebar"] * {{ color: var(--text); }}
        .sidebar-brand {{
            background: linear-gradient(135deg, rgba(124, 92, 255, 0.24), rgba(52, 211, 153, 0.10));
            border: 1px solid var(--soft-border); border-radius: 12px;
            box-shadow: var(--soft-shadow); padding: 20px 18px; margin: 0.25rem 0.15rem 1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.16);
        }}
        .sidebar-brand__mark {{
            align-items: center; background: #7c5cff; border-radius: 9px; display: inline-flex;
            font-size: 19px; height: 36px; justify-content: center; margin-right: 9px; width: 36px;
        }}
        .sidebar-brand__title {{ display: inline-block; font-size: 17px; font-weight: 800; letter-spacing: -0.02em; vertical-align: middle; }}
        .sidebar-brand__caption {{ color: var(--muted) !important; font-size: 12px; margin: 11px 0 0; line-height: 1.5; }}
        .nav-label {{ color: var(--muted) !important; font-size: 11px; font-weight: 700; letter-spacing: 0.10em; margin: 0 0 0.45rem 0.6rem; }}
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {{ gap: 0.35rem; }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label {{
            align-items: center; background: transparent; border: 1px solid transparent; border-radius: 8px;
            margin: 0; min-height: 43px; padding: 9px 10px; transition: all 0.18s ease;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background: rgba(255, 255, 255, 0.07); }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
            background: linear-gradient(90deg, rgba(124, 92, 255, 0.32), rgba(124, 92, 255, 0.10));
            border-color: rgba(167, 139, 250, 0.55); box-shadow: inset 3px 0 0 #a78bfa;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label p {{ font-size: 14px; font-weight: 600; margin: 0; }}

        [data-testid="stMetric"] {{
            background: linear-gradient(145deg, rgba(29, 39, 64, 0.96), rgba(22, 29, 49, 0.96));
            border: 1px solid var(--soft-border); border-radius: 12px; box-shadow: var(--soft-shadow);
            min-height: 128px; padding: 20px 24px;
        }}
        [data-testid="stMetricLabel"] p {{ color: var(--muted) !important; font-size: 14px !important; font-weight: 600; }}
        [data-testid="stMetricValue"] {{ color: var(--text) !important; font-size: 32px !important; font-weight: 800 !important; line-height: 1.2; }}
        [data-testid="stAlert"] {{
            background: rgba(29, 39, 64, 0.85); border: 1px solid var(--soft-border); border-radius: 12px;
            box-shadow: var(--soft-shadow); padding: 20px 24px;
        }}
        .stButton > button, [data-testid="stDownloadButton"] > button {{
            background: #26314d; border: 1px solid rgba(255,255,255,0.16); border-radius: 8px;
            box-shadow: 0 3px 9px rgba(0, 0, 0, 0.14); color: var(--text); font-size: 14px;
            font-weight: 600; min-height: 42px; padding: 10px 20px; transition: transform 0.18s ease, filter 0.18s ease, background 0.18s ease;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, #7358ee, #8b5cf6); border-color: #9b8afb; color: #ffffff;
        }}
        .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {{
            filter: brightness(1.14); transform: translateY(-2px);
        }}
        .stButton > button:active, [data-testid="stDownloadButton"] > button:active {{ transform: translateY(0); }}
        [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-testid="stFileUploader"] section {{
            background: rgba(22, 29, 49, 0.92) !important; border: 1px solid var(--soft-border) !important;
            border-radius: 8px !important; min-height: 42px; padding: 3px 7px;
        }}
        [data-testid="stFileUploader"] {{
            background: rgba(22, 29, 49, 0.65); border: 1px solid var(--soft-border); border-radius: 12px;
            box-shadow: var(--soft-shadow); padding: 10px;
        }}
        [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{ background: #a78bfa; }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            border: 1px solid var(--soft-border); border-radius: 12px; box-shadow: var(--soft-shadow); overflow: hidden;
        }}
        [data-testid="stTabs"] [data-baseweb="tab-list"] {{ border-bottom: 1px solid var(--soft-border); gap: 0.5rem; }}
        [data-testid="stTabs"] button {{ border-radius: 8px 8px 0 0; font-size: 14px; font-weight: 600; padding: 10px 16px; }}
        [data-testid="stTabs"] [aria-selected="true"] {{ color: #c4b5fd !important; }}
        [data-testid="stHorizontalBlock"] {{ gap: 1.25rem; margin-bottom: 1.15rem; }}
        @media (max-width: 768px) {{ .block-container {{ padding: 1.4rem 1rem 2.5rem; }} h1 {{ font-size: 26px !important; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_reference_layout_style() -> None:
    """Override the base theme with the supplied operations-dashboard layout."""
    st.markdown(
        """
        <style>
        .stApp { background: #f4f7fb !important; color: #1f2937 !important; }
        .block-container { max-width: none !important; padding: 1.1rem 1.45rem 2.2rem !important; }
        [data-testid="stHeader"] { background: #f8fbff !important; border-bottom: 1px solid #dfe8f4; }
        h1, h2, h3, p, [data-testid="stCaptionContainer"] { color: #1d2a3d !important; }
        h1 { font-size: 18px !important; font-weight: 700 !important; margin: 0.15rem 0 0.75rem !important; }
        h2, h3 { font-size: 13px !important; font-weight: 700 !important; margin: 0 0 0.65rem !important; }
        [data-testid="stSidebar"] { background: #0b203a !important; padding: 0.35rem 0.38rem !important; }
        [data-testid="stSidebar"] * { color: #dce9f8 !important; }
        /* Keep only the app's custom SEMICON AI navigation; hide Streamlit's pages/ navigator. */
        [data-testid="stSidebarNav"] { display: none !important; }
        .sidebar-brand { background: transparent !important; border: 0 !important; border-bottom: 1px solid rgba(255,255,255,.17) !important; border-radius: 0 !important; box-shadow: none !important; margin: 0 !important; padding: 0.7rem 0.7rem 0.85rem !important; }
        .sidebar-brand__mark { background: #1771cf !important; border-radius: 5px !important; font-size: 14px !important; height: 25px !important; width: 25px !important; }
        .sidebar-brand__title { font-size: 12px !important; letter-spacing: .02em !important; }
        .sidebar-brand__caption { color: #9cb5cf !important; font-size: 9px !important; line-height: 1.4 !important; margin-top: 6px !important; }
        .nav-label { color: #7894b0 !important; font-size: 9px !important; margin: 1rem 0 0.35rem .55rem !important; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { min-height: 30px !important; padding: 6px 8px !important; border-radius: 4px !important; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #12365e !important; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background: #1265b7 !important; border-color: #1b78d2 !important; box-shadow: none !important; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label p { color: #eaf3ff !important; font-size: 11px !important; font-weight: 600 !important; }
        .app-topbar { align-items: center; background: #ffffff; border: 1px solid #e1eaf4; border-radius: 5px; box-shadow: 0 1px 3px rgba(34,60,90,.05); display: flex; height: 37px; justify-content: space-between; margin: -0.35rem 0 0.9rem; padding: 0 0.8rem; }
        .app-topbar__crumb { color: #6f8299; font-size: 10px; }
        .app-topbar__search { background: #f6f9fd; border: 1px solid #e1e9f4; border-radius: 3px; color: #8a9bb0; font-size: 10px; padding: 5px 10px; width: 235px; }
        .app-topbar__user { color: #35516e; font-size: 10px; font-weight: 700; }
        .hero-panel { background: linear-gradient(100deg, #0c294c 0%, #164e88 55%, #357cbd 100%); border-radius: 5px; box-shadow: 0 3px 10px rgba(20,57,93,.16); color: #ffffff; min-height: 94px; overflow: hidden; padding: 18px 20px; position: relative; }
        .hero-panel:after { background: radial-gradient(circle, rgba(255,255,255,.26) 0, rgba(255,255,255,0) 62%); content: ''; height: 190px; position: absolute; right: -35px; top: -58px; width: 280px; }
        .hero-panel__eyebrow { color: #b9d9f7; font-size: 10px; font-weight: 600; letter-spacing: .08em; margin-bottom: 5px; text-transform: uppercase; }
        .hero-panel__title { color: #ffffff; font-size: 19px; font-weight: 700; position: relative; z-index: 1; }
        .hero-panel__copy { color: #d8eafa; font-size: 11px; margin-top: 5px; position: relative; z-index: 1; }
        .mini-status { background: #ffffff; border: 1px solid #dce7f2; border-radius: 5px; box-shadow: 0 2px 7px rgba(34,60,90,.08); min-height: 94px; overflow: hidden; }
        .dashboard-card-header, .dashboard-section-header { background: #0c4f8f; color: #ffffff !important; font-size: 11px; font-weight: 700; }
        .dashboard-card-header { padding: 8px 11px; }
        .mini-status__body { padding: 10px 14px 12px; }
        .mini-status__label { color: #74869a; font-size: 10px; font-weight: 600; }
        .mini-status__value { color: #183c60; font-size: 17px; font-weight: 800; margin-top: 6px; }
        .mini-status__meta { color: #20a36a; font-size: 10px; font-weight: 700; margin-top: 5px; }
        .dashboard-kpi { background: #ffffff; border: 1px solid #dfe8f2; border-radius: 5px; box-shadow: 0 2px 7px rgba(34,60,90,.07); min-height: 78px; overflow: hidden; }
        .dashboard-kpi__label { background: #0c4f8f; color: #ffffff; font-size: 10px; font-weight: 700; padding: 7px 10px; }
        .dashboard-kpi__value { color: #1b3b5f; font-size: 20px; font-weight: 800; line-height: 1.2; padding: 7px 10px 9px; }
        .dashboard-section-header { margin: -1rem -1rem .8rem; padding: 9px 12px; }
        .realtime-page-title { background: #1265b7; border-radius: 20px; color: #ffffff !important; display: inline-block; font-size: 16px; font-weight: 700; line-height: 1; margin: 0.45rem 0 1rem; padding: 10px 18px; }
        .realtime-section-label { background: #ffffff; border: 1px solid #1265b7; border-radius: 5px; color: #1265b7; display: inline-block; font-size: 11px; font-weight: 700; margin: 0.15rem 0 0.35rem; padding: 5px 14px; }
        .data-management-page-title { background: #1b65a5; color: #ffffff !important; display: inline-block; font-size: 15px; font-weight: 700; line-height: 1; margin: 0.45rem 0 1rem; padding: 9px 22px; }
        .dataset-category-header { background: #1b65a5; box-sizing: border-box; color: #ffffff !important; font-size: 13px; font-weight: 700; margin: 0.15rem 0 0.7rem; padding: 9px 14px; width: 100%; }
        .quality-report-title { background: #1b65a5; color: #ffffff !important; display: inline-block; font-size: 15px; font-weight: 700; line-height: 1; margin: 0.45rem 0 0.9rem; padding: 9px 22px; }
        .quality-report-section-label { background: #ffffff; border: 1px solid #1b65a5; border-radius: 5px; color: #1b65a5 !important; display: inline-block; font-size: 11px; font-weight: 700; line-height: 1; margin: 0.35rem 0 0.45rem; padding: 6px 16px; }
        .process-page-title { background: #1b65a5; color: #ffffff !important; display: inline-block; font-size: 15px; font-weight: 700; line-height: 1; margin: 0.45rem 0 0.9rem; padding: 9px 22px; }
        .process-section-label { background: #ffffff; border: 1px solid #1b65a5; border-radius: 5px; color: #1b65a5 !important; display: inline-block; font-size: 11px; font-weight: 700; line-height: 1; margin: 0.35rem 0 0.45rem; padding: 6px 16px; }
        .process-kpi { background: #ffffff; border: 1px solid #dfe8f2; border-radius: 5px; box-shadow: 0 2px 7px rgba(34,60,90,.07); min-height: 78px; overflow: hidden; }
        .process-kpi__label { background: #0c4f8f; color: #ffffff !important; font-size: 10px; font-weight: 700; padding: 7px 10px; }
        .process-kpi__value { color: #1b3b5f; font-size: 20px; font-weight: 800; line-height: 1.2; padding: 7px 10px 9px; }
        [data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] { gap: 0.45rem; }
        [data-testid="stMain"] [data-testid="stRadio"] label { align-items: center; background: #1265b7; border: 1px solid #1265b7; border-radius: 999px; color: #ffffff !important; justify-content: center; margin: 0; min-height: 25px; min-width: 130px; padding: 5px 14px; }
        [data-testid="stMain"] [data-testid="stRadio"] label:hover { background: #0c4f8f; border-color: #0c4f8f; }
        [data-testid="stMain"] [data-testid="stRadio"] label:has(input:checked) { background: #0c4f8f; border-color: #0c4f8f; box-shadow: inset 0 0 0 1px rgba(255,255,255,.5); }
        [data-testid="stMain"] [data-testid="stRadio"] label > div:first-child { display: none; }
        [data-testid="stMain"] [data-testid="stRadio"] label p { color: #ffffff !important; font-size: 11px !important; font-weight: 700; margin: 0; }
        [data-testid="stMain"] [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background: #1265b7 !important; border-color: #ffffff !important; box-shadow: 0 0 0 1px #1265b7; }
        [data-testid="stMain"] [data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: #1265b7 !important; }
        [data-testid="stMain"] [data-testid="stFileUploader"] { background: rgba(81, 94, 117, .88); border: 0; border-radius: 8px; box-shadow: 0 3px 9px rgba(34,60,90,.16); padding: 10px; }
        [data-testid="stMain"] [data-testid="stFileUploader"] label { color: #eaf3ff !important; font-size: 11px !important; font-weight: 600; }
        [data-testid="stMetric"] { background: #ffffff !important; border: 1px solid #dfe8f2 !important; border-radius: 5px !important; box-shadow: 0 2px 7px rgba(34,60,90,.07) !important; min-height: 78px !important; padding: 12px 14px !important; }
        [data-testid="stMetricLabel"] p { color: #70839a !important; font-size: 10px !important; font-weight: 600 !important; }
        [data-testid="stMetricValue"] { color: #1b3b5f !important; font-size: 20px !important; font-weight: 800 !important; }
        .dashboard-panel { background: #ffffff; border: 1px solid #dfe8f2; border-radius: 5px; box-shadow: 0 2px 7px rgba(34,60,90,.07); padding: 13px 14px 7px; }
        .panel-title { color: #254868; font-size: 12px; font-weight: 700; margin: 0 0 8px; }
        [data-testid="stVerticalBlockBorderWrapper"] { background: #ffffff; border: 1px solid #dfe8f2 !important; border-radius: 5px !important; box-shadow: 0 2px 7px rgba(34,60,90,.07); }
        [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #dfe8f2 !important; border-radius: 5px !important; box-shadow: none !important; }
        .stButton > button, [data-testid="stDownloadButton"] > button { background: #ffffff !important; border: 1px solid #cddcea !important; border-radius: 4px !important; box-shadow: none !important; color: #2863a1 !important; font-size: 11px !important; min-height: 30px !important; padding: 5px 11px !important; }
        .stButton > button[kind="primary"] { background: #1467b8 !important; border-color: #1467b8 !important; color: #ffffff !important; }
        .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover { filter: brightness(1.05); transform: translateY(-1px); }
        [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-testid="stFileUploader"] section { background: #ffffff !important; border-color: #d8e3ee !important; border-radius: 4px !important; }
        [data-testid="stAlert"] { background: #ffffff !important; border: 1px solid #dfe8f2 !important; border-radius: 5px !important; box-shadow: 0 2px 7px rgba(34,60,90,.06) !important; color: #30506e !important; padding: 12px 14px !important; }
        .judgement-badge { border-radius: 999px; display: inline-block; font-size: 13px; font-weight: 800; letter-spacing: -.01em; margin: 0.1rem 0 0.85rem; padding: 9px 13px; }
        .judgement-badge--pass { background: #e9f9f0; border: 1px solid #96ddaf; color: #158347; }
        .judgement-badge--ng { background: #fff0f1; border: 1px solid #f2a3aa; color: #cf3140; }
        .inspection-meta { background: #f6f9fc; border: 1px solid #e0e8f1; border-radius: 5px; color: #4b647e; font-size: 11px; line-height: 1.8; margin: 0.65rem 0; padding: 9px 11px; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { border-bottom: 1px solid #dce7f2 !important; }
        [data-testid="stTabs"] button { color: #60778e !important; font-size: 11px !important; padding: 8px 12px !important; }
        [data-testid="stTabs"] [aria-selected="true"] { color: #1265b7 !important; }
        [data-testid="stHorizontalBlock"] { gap: .7rem !important; margin-bottom: .7rem !important; }
        @media (max-width: 800px) { .block-container { padding: .8rem !important; } .app-topbar__search { display: none; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_background_image() -> None:
    """Use the project's background asset behind every Streamlit view."""
    if not BACKGROUND_IMAGE_PATH.exists():
        return

    image_base64 = base64.b64encode(BACKGROUND_IMAGE_PATH.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(244, 247, 251, 0.30), rgba(244, 247, 251, 0.30)),
                url("data:image/png;base64,{image_base64}") !important;
            background-attachment: fixed !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-size: cover !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_topbar() -> None:
    """Render the fixed visual shell used by every page."""
    st.markdown(
        """
        <div class="app-topbar">
            <span class="app-topbar__crumb">운영 대시보드 / Vision Q</span>
            <span class="app-topbar__user">● 품질 검사 시스템</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_db() -> None:
    """Create or upgrade the inspection schema for every selectable model."""
    # 모든 화면이 공유하는 검사 이력 저장소를 앱 시작 시 보장한다.
    allowed_models = ", ".join(f"'{model_id}'" for model_id in MODEL_IDENTIFIERS)
    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            result TEXT NOT NULL CHECK (result IN ('{RESULT_NORMAL}', '{RESULT_DEFECTIVE}')),
            defect_count INTEGER NOT NULL CHECK (defect_count >= 0),
            model_used TEXT NOT NULL CHECK (
                model_used IN ({allowed_models})
            ),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    with sqlite3.connect(DB_PATH) as connection:
        schema_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'inspections'"
        ).fetchone()
        schema_sql = schema_row[0] if schema_row else ""
        if schema_row and any(model_id not in schema_sql for model_id in MODEL_IDENTIFIERS):
            connection.execute("ALTER TABLE inspections RENAME TO inspections_legacy")
            connection.execute(create_table_sql)
            connection.execute(
                """
                INSERT INTO inspections (id, image_name, result, defect_count, model_used, created_at)
                SELECT id, image_name, result, defect_count, model_used, created_at
                FROM inspections_legacy
                """
            )
            connection.execute("DROP TABLE inspections_legacy")
        else:
            connection.execute(create_table_sql)
        # Earlier builds created only these known fake rows. Keep real inference records.
        connection.execute("DELETE FROM inspections WHERE image_name LIKE ?", ("sample_wafer_%",))


def load_inspections() -> pd.DataFrame:
    """Return persisted, real inference history newest first."""
    query = """
        SELECT id, image_name, result, defect_count, model_used, created_at
        FROM inspections
        ORDER BY datetime(created_at) DESC, id DESC
    """
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(query, connection)


def save_inspection(
    image_name: str, result: str, defect_count: int, model_used: str
) -> None:
    """Persist a model prediction with parameterized SQL."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO inspections (image_name, result, defect_count, model_used)
            VALUES (?, ?, ?, ?)
            """,
            (image_name, result, defect_count, model_used),
        )


def save_inspections(records: list[tuple[str, str, int, str]]) -> None:
    """Persist an inference batch in one SQLite transaction."""
    # 배치 추론 결과를 한 트랜잭션으로 기록해 이력과 대시보드 집계를 동기화한다.
    with sqlite3.connect(DB_PATH) as connection:
        connection.executemany(
            """
            INSERT INTO inspections (image_name, result, defect_count, model_used)
            VALUES (?, ?, ?, ?)
            """,
            records,
        )


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode an uploaded image into an RGB ndarray for YOLO."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("RGB"))


def load_uploaded_images(uploaded_files) -> tuple[list[tuple[str, np.ndarray]], list[str]]:
    """Load direct image uploads and image entries from ZIP uploads in memory."""
    # 이미지와 ZIP 입력을 모두 RGB 배열로 표준화하여 동일한 YOLO 추론 경로로 전달한다.
    images: list[tuple[str, np.ndarray]] = []
    errors: list[str] = []
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            try:
                images.append((uploaded_file.name, decode_image(uploaded_file.getvalue())))
            except (OSError, ValueError) as error:
                errors.append(f"{uploaded_file.name}: 이미지를 읽지 못했습니다 ({error}).")
        elif suffix == ".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as archive:
                    for entry in archive.infolist():
                        entry_path = Path(entry.filename)
                        if entry.is_dir() or entry_path.suffix.lower() not in IMAGE_EXTENSIONS:
                            continue
                        try:
                            image_name = f"{uploaded_file.name}/{entry_path.name}"
                            images.append((image_name, decode_image(archive.read(entry))))
                        except (OSError, ValueError, zipfile.BadZipFile) as error:
                            errors.append(f"{uploaded_file.name}/{entry.filename}: 읽기 실패 ({error}).")
            except zipfile.BadZipFile:
                errors.append(f"{uploaded_file.name}: 올바른 ZIP 파일이 아닙니다.")
    return images, errors


def build_detection_details(prediction) -> pd.DataFrame:
    """Convert YOLO boxes into a readable object-level inspection table."""
    # YOLO의 box 텐서를 사용자가 확인할 수 있는 클래스·신뢰도·좌표 표로 변환한다.
    label_map = {"bad": "불량", "good": "정상"}
    rows = []
    if prediction.boxes is None:
        return pd.DataFrame(columns=["No.", "결함 라벨", "Confidence(%)", "위치 정보"])
    for number, (class_id, confidence, coordinates) in enumerate(
        zip(
            prediction.boxes.cls.tolist(),
            prediction.boxes.conf.tolist(),
            prediction.boxes.xyxy.tolist(),
        ),
        start=1,
    ):
        english_label = str(prediction.names[int(class_id)])
        korean_label = label_map.get(english_label.lower(), english_label)
        x1, y1, x2, y2 = (round(value) for value in coordinates)
        rows.append(
            {
                "No.": number,
                "결함 라벨": f"{korean_label} ({english_label})",
                "Confidence(%)": round(float(confidence) * 100, 1),
                "위치 정보": f"({x1}, {y1}) – ({x2}, {y2})",
            }
        )
    return pd.DataFrame(rows)


def render_judgement_badge(inspection_result: str) -> None:
    """Render a high-visibility PASS/NG badge for an inspection result."""
    if inspection_result == RESULT_DEFECTIVE:
        content = "🔴 판정: 불량 (NG)"
        modifier = "ng"
    else:
        content = "🟢 판정: 정상 (PASS)"
        modifier = "pass"
    st.markdown(
        f'<div class="judgement-badge judgement-badge--{modifier}">{content}</div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="YOLO 모델을 불러오는 중입니다...")
def load_yolo_model(model_path: str):
    """Load an actual project weight file once per session."""
    # 무거운 가중치는 모델별로 한 번만 메모리에 올려 반복 추론 지연을 줄인다.
    from ultralytics import YOLO

    return YOLO(model_path)


def get_images(directory: Path, extensions: set[str]) -> list[Path]:
    """Return image assets from one project directory."""
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def get_sample_images() -> list[Path]:
    return get_images(SAMPLE_IMAGE_DIR, {".jpg", ".jpeg", ".png", ".bmp"})


def load_last_epoch_metrics(
    model_id: str, model_analysis: dict[str, object]
) -> dict[str, object] | None:
    """Read final metrics from the selected model's configured results.csv."""
    # 학습 CSV의 마지막 epoch를 PRD의 모델 성능 비교용 지표로 정규화한다.
    csv_path = Path(model_analysis["metrics_csv"])
    if not csv_path.exists():
        return None
    results = pd.read_csv(csv_path)
    if results.empty:
        return None
    results.columns = results.columns.str.strip()
    last_epoch = results.iloc[-1]
    return {
        "모델": model_analysis["display_name"],
        "Epoch": int(last_epoch["epoch"]),
        "Precision": float(last_epoch["metrics/precision(B)"]),
        "Recall": float(last_epoch["metrics/recall(B)"]),
        "mAP50": float(last_epoch["metrics/mAP50(B)"]),
        "mAP50-95": float(last_epoch["metrics/mAP50-95(B)"]),
    }


def apply_chart_theme(figure):
    """Apply the light operations-dashboard chart appearance."""
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#48637f", "family": "Inter, Noto Sans KR, sans-serif", "size": 10},
        margin={"l": 10, "r": 10, "t": 18, "b": 10},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 9}},
        xaxis={"gridcolor": "#edf2f7", "linecolor": "#dfe8f2"},
        yaxis={"gridcolor": "#edf2f7", "linecolor": "#dfe8f2"},
    )
    return figure


def render_main_dashboard() -> None:
    """Render the reference-image dashboard grid using persisted inspection data."""
    # SQLite에 누적된 검사 이력을 KPI, 추세, 최근 검사 목록으로 요약한다.
    render_topbar()
    inspections = load_inspections()
    total_count = len(inspections)
    normal_count = int((inspections["result"] == RESULT_NORMAL).sum())
    defective_count = int((inspections["result"] == RESULT_DEFECTIVE).sum())
    defect_rate = defective_count / total_count * 100 if total_count else 0
    available_models = sum(
        (PROJECT_ROOT / path).exists() for path in MODEL_OPTIONS.values()
    )

    hero_column, status_column = st.columns((3.1, 1))
    with hero_column:
        st.markdown(
            """
            <div class="hero-panel">
                <div class="hero-panel__eyebrow">QUALITY OPERATIONS CENTER</div>
                <div class="hero-panel__title">결함 검출 및 통합 분석 현황</div>
                <div class="hero-panel__copy">프로젝트 자산과 실시간 검사 이력을 한 화면에서 확인합니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_column:
        st.markdown(
            f"""
            <div class="mini-status">
                <div class="dashboard-card-header">모델 운영 상태</div>
                <div class="mini-status__body">
                    <div class="mini-status__value">{available_models} / {len(MODEL_OPTIONS)} Ready</div>
                    <div class="mini-status__meta">● 프로젝트 모델 연결됨</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    cards = st.columns(4)
    for column, label, value in zip(
        cards,
        ["전체 검사 수", "정상", "불량", "불량률"],
        [f"{total_count:,}건", f"{normal_count:,}건", f"{defective_count:,}건", f"{defect_rate:.1f}%"],
    ):
        with column:
            st.markdown(
                f"""
                <div class="dashboard-kpi">
                    <div class="dashboard-kpi__label">{label}</div>
                    <div class="dashboard-kpi__value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if inspections.empty:
        st.info("저장된 실제 검사 이력이 없습니다. ‘실시간 결함 검사’에서 첫 검사를 실행하세요.")
        return

    trend_column, result_column, history_column = st.columns((1.35, 1.05, 1.6))
    with trend_column:
        with st.container(border=True):
            st.markdown('<div class="dashboard-section-header">검사 이력 추이</div>', unsafe_allow_html=True)
            activity = inspections.copy()
            activity["date"] = pd.to_datetime(activity["created_at"]).dt.date
            daily_activity = activity.groupby(["date", "result"]).size().reset_index(name="count")
            trend = px.bar(
                daily_activity,
                x="date",
                y="count",
                color="result",
                barmode="group",
                color_discrete_map={RESULT_NORMAL: "#4a90e2", RESULT_DEFECTIVE: "#f06a6a"},
                labels={"date": "일자", "count": "검사 건수", "result": "판정"},
            )
            st.plotly_chart(apply_chart_theme(trend), width="stretch")

    with result_column:
        with st.container(border=True):
            st.markdown('<div class="dashboard-section-header">검사 결과 현황</div>', unsafe_allow_html=True)
            result_counts = (
                inspections["result"].value_counts()
                .reindex([RESULT_NORMAL, RESULT_DEFECTIVE], fill_value=0)
                .rename_axis("result").reset_index(name="count")
            )
            donut = px.pie(
                result_counts,
                names="result",
                values="count",
                color="result",
                color_discrete_map={RESULT_NORMAL: "#4a90e2", RESULT_DEFECTIVE: "#f06a6a"},
                hole=0.58,
            )
            donut.update_traces(textinfo="label+percent", textfont_size=10)
            st.plotly_chart(apply_chart_theme(donut), width="stretch")

    with history_column:
        with st.container(border=True):
            st.markdown('<div class="dashboard-section-header">최근 검사 결과</div>', unsafe_allow_html=True)
            st.dataframe(inspections.head(5), width="stretch", hide_index=True, height=265)


def render_realtime_inspection() -> None:
    """Run single or batch YOLO inference from uploads, ZIPs, or test assets."""
    # 업로드·샘플·빠른 샘플 입력을 하나의 실시간 YOLO 검사 흐름으로 통합한다.
    st.markdown('<div class="realtime-page-title">실시간 결함 검사</div>', unsafe_allow_html=True)
    st.markdown('<div class="realtime-section-label">검사 모델 선택</div>', unsafe_allow_html=True)
    model_path = st.radio(
        "검사 모델 선택",
        list(MODEL_OPTIONS),
        index=2,
        horizontal=True,
        label_visibility="collapsed",
    )
    model_path = MODEL_OPTIONS[model_path]
    st.markdown('<div class="realtime-section-label">불량 감지 정도 (0.50 추천)</div>', unsafe_allow_html=True)
    confidence_threshold = st.slider(
        "불량 감지할 정도를 정해주세요 (0.50 추천)", 0.01, 1.00, 0.25, 0.01,
        help="이 값 이상인 검출만 최종 판정에 반영합니다.",
        label_visibility="collapsed",
    )
    st.markdown('<div class="realtime-section-label">이미지 입력 방식</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "이미지 입력 방식",
        ["직접 파일 업로드", "테스트 이미지 선택"],
        horizontal=True,
        label_visibility="collapsed",
    )
    sample_images = get_sample_images()
    quick_sample_clicked = st.button(
        "⚡ Quick 샘플 즉시 검사", help="프로젝트 테스트 이미지 한 장으로 바로 검사합니다."
    )

    image_inputs: list[tuple[str, np.ndarray]] = []
    run_requested = quick_sample_clicked
    if quick_sample_clicked:
        if sample_images:
            quick_image = sample_images[0]
            image_inputs = [(quick_image.name, np.asarray(Image.open(quick_image).convert("RGB")))]
            st.info(f"Quick 샘플: {quick_image.name}")
        else:
            st.toast("Quick 샘플로 사용할 테스트 이미지를 찾을 수 없습니다.", icon="⚠️")
            run_requested = False
    elif input_mode == "직접 파일 업로드":
        uploaded_files = st.file_uploader(
            "검사할 이미지 또는 ZIP 파일 업로드",
            type=["jpg", "jpeg", "png", "bmp", "zip"],
            accept_multiple_files=True,
            help="여러 이미지 또는 이미지가 포함된 ZIP 파일을 함께 업로드할 수 있습니다.",
        )
        if uploaded_files:
            image_inputs, load_errors = load_uploaded_images(uploaded_files)
            st.info(f"총 {len(image_inputs)}장의 이미지가 로드되었습니다.")
            for error_message in load_errors:
                st.warning(error_message)
    elif sample_images:
        selected_image = st.selectbox(
            "테스트 이미지 선택", sample_images, format_func=lambda path: path.name
        )
        image_inputs = [(selected_image.name, np.asarray(Image.open(selected_image).convert("RGB")))]
    else:
        st.warning(f"테스트 이미지 폴더를 찾을 수 없습니다: {SAMPLE_IMAGE_DIR}")

    if len(image_inputs) == 1:
        st.image(image_inputs[0][1], caption=f"입력 이미지: {image_inputs[0][0]}", width=450)
    elif len(image_inputs) > 1:
        st.caption(f"배치 검사 대기 중: {len(image_inputs)}장")

    run_requested = st.button("검사 실행", type="primary") or run_requested
    if not run_requested:
        return
    if not image_inputs:
        st.toast("검사할 이미지 또는 ZIP 파일을 업로드하세요.", icon="⚠️")
        return

    resolved_model_path = PROJECT_ROOT / model_path
    if not resolved_model_path.exists():
        st.error(f"모델 파일을 찾을 수 없습니다: {resolved_model_path}")
        return

    try:
        with st.spinner("YOLO 모델을 준비하는 중입니다..."):
            model = load_yolo_model(str(resolved_model_path))
        progress = st.progress(0, text=f"0 / {len(image_inputs)} 이미지 검사 준비 중")
        batch_results = []
        database_records = []
        model_used = Path(model_path).stem
        for index, (image_name, image_array) in enumerate(image_inputs, start=1):
            # 'bad' 클래스가 하나라도 검출되면 불량으로 판정하고, 결과 이미지는 즉시 표시용으로 보관한다.
            prediction = model.predict(
                image_array, conf=confidence_threshold, verbose=False
            )[0]
            class_ids = prediction.boxes.cls.tolist() if prediction.boxes else []
            defect_count = sum(
                str(prediction.names[int(class_id)]).lower() == "bad"
                for class_id in class_ids
            )
            inspection_result = RESULT_DEFECTIVE if defect_count else RESULT_NORMAL
            database_records.append((image_name, inspection_result, defect_count, model_used))
            batch_results.append(
                {
                    "image_name": image_name,
                    "result": inspection_result,
                    "defect_count": defect_count,
                    "original_image": image_array,
                    "annotated_image": prediction.plot(),
                    "details": build_detection_details(prediction),
                    "inference_ms": float(prediction.speed.get("inference", 0.0)),
                }
            )
            progress.progress(
                index / len(image_inputs),
                text=f"{index} / {len(image_inputs)} 이미지 검사 중: {image_name}",
            )

        # 화면 표시 전에 전체 배치의 메타데이터만 DB에 저장한다(원본 이미지는 저장하지 않음).
        save_inspections(database_records)
        progress.empty()
        st.success(f"{len(batch_results)}장 검사 완료 · 모든 결과를 SQLite 이력에 저장했습니다.")

        if len(batch_results) == 1:
            result = batch_results[0]
            result_column, summary_column = st.columns((2, 1))
            with result_column:
                result_tab, original_tab = st.tabs(["결함 탐지 결과", "원본 이미지"])
                with result_tab:
                    st.image(
                        result["annotated_image"],
                        channels="BGR",
                        caption="Bounding Box가 표시된 YOLO 추론 결과",
                    )
                with original_tab:
                    st.image(result["original_image"], caption="업로드된 원본 이미지")
            with summary_column:
                st.subheader("판정 요약")
                render_judgement_badge(result["result"])
                st.metric("불량 개수", f"{result['defect_count']}개")
                st.markdown(
                    f"""
                    <div class="inspection-meta">
                    <b>모델</b> {model_used}<br>
                    <b>Threshold</b> {confidence_threshold:.2f}<br>
                    <b>Inference</b> {result['inference_ms']:.1f} ms
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("검출 결함 상세")
                if result["details"].empty:
                    st.info("Threshold 이상으로 검출된 객체가 없습니다.")
                else:
                    st.dataframe(result["details"], width="stretch", hide_index=True, height=220)
        else:
            st.subheader("배치 검사 결과")
            for start_index in range(0, len(batch_results), 3):
                gallery_columns = st.columns(3)
                for column, result in zip(gallery_columns, batch_results[start_index:start_index + 3]):
                    with column:
                        st.image(result["annotated_image"], channels="BGR", width="stretch")
                        st.caption(result["image_name"])
                        render_judgement_badge(result["result"])
                        st.caption(
                            f"Detect 수: {result['defect_count']} · "
                            f"Inference: {result['inference_ms']:.1f} ms"
                        )
    except Exception as error:
        st.error(f"검사 중 오류가 발생했습니다: {error}")


def render_data_training_management() -> None:
    """Render the dataset and one configurable training-analysis tab per model."""
    # 실제 데이터셋과 각 YOLO 실험 산출물을 한 화면에서 비교·점검한다.
    st.markdown('<div class="data-management-page-title">데이터 및 학습 관리</div>', unsafe_allow_html=True)
    tab_labels = ["데이터셋 현황"] + [
        analysis["display_name"] for analysis in MODEL_ANALYSES.values()
    ]
    tabs = st.tabs(tab_labels)
    dataset_tab = tabs[0]
    with dataset_tab:
        sources = {
            "불량 데이터": (BAD_DATASET_DIR, {".bmp"}),
            "정상 데이터": (GOOD_DATASET_DIR, {".jpg", ".jpeg"}),
        }
        for column, (title, (directory, extensions)) in zip(st.columns(2), sources.items()):
            images = get_images(directory, extensions)
            with column:
                st.markdown(f'<div class="dataset-category-header">{title}</div>', unsafe_allow_html=True)
                st.metric("이미지 수", f"{len(images):,}장")
                if images:
                    preview_columns = st.columns(2)
                    for index, image_path in enumerate(images[:4]):
                        with preview_columns[index % 2]:
                            st.image(image_path, caption=image_path.name, width="stretch")
                else:
                    st.warning(f"이미지가 없거나 폴더를 찾을 수 없습니다: {directory}")

    for tab, (model_id, model_analysis) in zip(tabs[1:], MODEL_ANALYSES.items()):
        with tab:
            model_name = str(model_analysis["display_name"])
            st.subheader("학습 결과 차트 (Results)")
            results_chart = Path(model_analysis["results_chart"])
            if results_chart.exists():
                st.image(
                    results_chart,
                    caption=f"{model_name} · results.png",
                    width="stretch",
                )
            else:
                st.warning(f"자산을 찾을 수 없습니다: {results_chart}")

            st.subheader("Confusion Matrix")
            confusion_candidates = [
                Path(model_analysis["confusion_matrix"]),
                Path(model_analysis["normalized_confusion_matrix"]),
            ]
            confusion_matrix = next(
                (path for path in confusion_candidates if path.exists()), None
            )
            if confusion_matrix is not None:
                st.image(
                    confusion_matrix,
                    caption=f"{model_name} · {confusion_matrix.name}",
                    width="stretch",
                )
            else:
                st.warning(
                    f"혼동 행렬 자산을 찾을 수 없습니다: {confusion_candidates[0]}"
                )

            with st.expander("성능 곡선 상세", expanded=False):
                curve_directory = Path(model_analysis["metrics_csv"]).parent
                curve_paths = [
                    curve_directory / filename
                    for filename in model_analysis["performance_curves"]
                ]
                for start_index in range(0, len(curve_paths), 2):
                    for column, curve_path in zip(
                        st.columns(2), curve_paths[start_index:start_index + 2]
                    ):
                        with column:
                            if curve_path.exists():
                                st.image(curve_path, caption=curve_path.name, width="stretch")
                            else:
                                st.caption(f"누락됨: {curve_path.name}")

            st.subheader("마지막 Epoch 성능 요약")
            try:
                metrics = load_last_epoch_metrics(model_id, model_analysis)
            except (KeyError, pd.errors.ParserError) as error:
                st.warning(f"{model_name} results.csv를 읽지 못했습니다: {error}")
                metrics = None
            if metrics:
                st.dataframe(
                    pd.DataFrame([metrics]).style.format(
                        {
                            "Precision": "{:.4f}",
                            "Recall": "{:.4f}",
                            "mAP50": "{:.4f}",
                            "mAP50-95": "{:.4f}",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("표시할 실제 results.csv 성능 데이터가 없습니다.")


def render_quality_analysis_report() -> None:
    """Render a filterable report of real persisted inspection records."""
    # 저장된 검사 이력을 필터링하고 동일한 범위를 표·차트·CSV로 제공한다.
    st.markdown(
        '<div class="quality-report-title">품질 분석 리포트</div>',
        unsafe_allow_html=True,
    )
    inspections = load_inspections()
    st.markdown(
        '<div class="quality-report-section-label">검사 결과 필터</div>',
        unsafe_allow_html=True,
    )
    result_filter = st.selectbox(
        "검사 결과 필터",
        ["전체", RESULT_NORMAL, RESULT_DEFECTIVE],
        label_visibility="collapsed",
    )
    filtered = inspections if result_filter == "전체" else inspections.loc[inspections["result"] == result_filter]
    st.markdown(
        '<div class="quality-report-section-label">검사 이력</div>',
        unsafe_allow_html=True,
    )
    if filtered.empty:
        st.info("표시할 실제 검사 이력이 없습니다.")
    else:
        st.dataframe(filtered, width="stretch", hide_index=True)

    st.markdown(
        '<div class="quality-report-section-label">일별 불량 건수</div>',
        unsafe_allow_html=True,
    )
    defects = inspections.loc[inspections["result"] == RESULT_DEFECTIVE, ["created_at"]].copy()
    if defects.empty:
        st.info("표시할 불량 검사 이력이 없습니다.")
    else:
        defects["date"] = pd.to_datetime(defects["created_at"]).dt.date
        daily_defects = defects.groupby("date").size().reset_index(name="defect_count")
        figure = px.bar(
            daily_defects.sort_values("date"), x="date", y="defect_count",
            labels={"date": "날짜", "defect_count": "불량 건수"},
            color_discrete_sequence=[COLORS["defective"]],
        )
        st.plotly_chart(apply_chart_theme(figure), width="stretch")

    st.download_button(
        "필터링된 이력 CSV 다운로드",
        data=filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="inspection_report.csv",
        mime="text/csv",
        disabled=filtered.empty,
    )


@st.cache_resource
def get_process_review_logic():
    """Load the UI-free Process data helpers from the added pages asset."""
    # 페이지 UI와 분리된 공정 메타데이터 로직을 동적으로 불러와 기존 자산을 재사용한다.
    if not PROCESS_REVIEW_LOGIC_PATH.exists():
        raise FileNotFoundError(f"Process logic file not found: {PROCESS_REVIEW_LOGIC_PATH}")
    spec = importlib.util.spec_from_file_location(
        "semicon_process_review_logic", PROCESS_REVIEW_LOGIC_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load Process review logic module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_data
def load_process_metadata_for_app() -> tuple[dict, str | None]:
    """Load real process metadata using the added Process backend helpers."""
    return get_process_review_logic().load_process_metadata()


def rebuild_process_metadata() -> None:
    """Run the existing metadata builder and write its standard JSON output."""
    # 기존 빌더를 실행해 공정 메타데이터 JSON을 갱신한 뒤 캐시를 무효화한다.
    builder_path = PROJECT_ROOT / "build_process_metadata.py"
    spec = importlib.util.spec_from_file_location("semicon_process_builder", builder_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load Process metadata builder.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = module.build_metadata()
    module.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    module.OUTPUT_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    load_process_metadata_for_app.clear()


def render_process_metadata() -> None:
    """Render Process metadata with the existing dashboard visual system."""
    # 생성된 공정 메타데이터의 현황, 필터 결과, 데이터 품질 경고를 표시한다.
    st.markdown('<div class="process-page-title">공정 메타데이터</div>', unsafe_allow_html=True)
    action_column, generated_column = st.columns((1, 4))
    with action_column:
        if st.button("메타데이터 재생성"):
            try:
                with st.spinner("공정 이미지 메타데이터를 생성하는 중입니다..."):
                    rebuild_process_metadata()
                st.success("공정 메타데이터를 새로 생성했습니다.")
            except (OSError, ImportError, FileNotFoundError) as error:
                st.error(f"메타데이터 생성에 실패했습니다: {error}")

    try:
        process_logic = get_process_review_logic()
        metadata, load_error = load_process_metadata_for_app()
    except (ImportError, FileNotFoundError) as error:
        st.error(f"Process 기능을 불러오지 못했습니다: {error}")
        return
    if load_error:
        st.error(load_error)
        return

    generated_at = str(metadata.get("generated_at", "-"))
    with generated_column:
        st.caption(f"메타데이터 생성 시각: {generated_at}")
    records = process_logic.records_from(metadata)
    counts = process_logic.overview_counts(metadata, records)
    kpi_values = [
        ("Dataset Images", counts["dataset_images"]),
        ("Process Records", counts["process_records"]),
        ("Machines", counts["machines"]),
        ("Parse Errors", counts["parse_errors"]),
        ("Data Warnings", counts["data_warnings"]),
    ]
    for column, (label, value) in zip(st.columns(5), kpi_values):
        with column:
            st.markdown(
                f"""
                <div class="process-kpi">
                    <div class="process-kpi__label">{label}</div>
                    <div class="process-kpi__value">{value:,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="process-section-label">공정 레코드 필터</div>', unsafe_allow_html=True)
    options = process_logic.filter_options(records)
    machine_column, date_column, shift_column = st.columns(3)
    with machine_column:
        machine = st.selectbox("설비", options["machine"], key="process_machine")
    with date_column:
        date = st.selectbox("날짜", options["date"], key="process_date")
    with shift_column:
        shift = st.selectbox("교대", options["shift"], key="process_shift")
    filtered_records = process_logic.filter_records(
        records, {"machine": machine, "date": date, "shift": shift}
    )

    st.markdown('<div class="process-section-label">공정 레코드</div>', unsafe_allow_html=True)
    st.caption(f"필터 결과: {len(filtered_records):,}건")
    st.dataframe(
        pd.DataFrame(process_logic.table_rows(filtered_records)),
        width="stretch",
        hide_index=True,
        height=420,
    )

    st.markdown('<div class="process-section-label">데이터 품질 경고</div>', unsafe_allow_html=True)
    warning_data = process_logic.warning_rows(metadata)
    if warning_data:
        st.warning(f"데이터 일관성 경고 {len(warning_data):,}건이 있습니다.")
        st.dataframe(
            pd.DataFrame(warning_data), width="stretch", hide_index=True, height=220
        )
    else:
        st.success("현재 데이터 품질 경고가 없습니다.")


def main() -> None:
    st.set_page_config(page_title="VISION Q", page_icon="🔍", layout="wide")
    init_db()
    apply_custom_style()
    apply_reference_layout_style()
    apply_background_image()
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <span class="sidebar-brand__mark">◈</span><span class="sidebar-brand__title">VISION Q</span>
            <p class="sidebar-brand__caption">실제 프로젝트 자산 기반<br>제품 품질 검사 시스템</p>
        </div>
        <p class="nav-label">WORKSPACE</p>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "메뉴",
        [
            "1. 메인 대시보드",
            "2. 실시간 결함 검사",
            "3. 데이터 및 학습 관리",
            "4. 품질 분석 리포트",
            "5. 공정 메타데이터",
        ],
        format_func=lambda item: {
            "1. 메인 대시보드": "대시보드",
            "2. 실시간 결함 검사": "실시간 검사",
            "3. 데이터 및 학습 관리": "데이터 및 학습 관리",
            "4. 품질 분석 리포트": "품질 리포트",
            "5. 공정 메타데이터": "프로세스 검토",
        }[item],
    )
    if page == "1. 메인 대시보드":
        render_main_dashboard()
    elif page == "2. 실시간 결함 검사":
        render_topbar()
        render_realtime_inspection()
    elif page == "3. 데이터 및 학습 관리":
        render_topbar()
        render_data_training_management()
    elif page == "4. 품질 분석 리포트":
        render_topbar()
        render_quality_analysis_report()
    elif page == "5. 공정 메타데이터":
        render_topbar()
        render_process_metadata()


if __name__ == "__main__":
    main()
