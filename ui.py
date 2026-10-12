#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui.py -- Gradio Clinical Web Interface

Two-level cancer screening interface with clean model selector,
dedicated model performance & accuracy tab, and bilingual (EN/TH)
clinical output. Designed with a formal, professional medical aesthetic.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from config import (
    CLASS_CODES,
    CLASS_NAMES_EN,
    CLASS_NAMES_TH,
    CLASS_DESCRIPTIONS_EN,
    CLASS_DESCRIPTIONS_TH,
    CANCER_INDICES,
    BENIGN_INDICES,
    MODEL_REGISTRY,
    MODEL_NAMES,
    NUM_CLASSES,
)
from data import val_test_transform
from models import build_model
from evaluate import load_all_metrics


# =============================================================================
# CSS Stylesheet
# =============================================================================

GRADIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Sarabun:wght@300;400;500;600;700&display=swap');

:root {
    --slate-950: #020617;
    --slate-900: #0f172a;
    --slate-800: #1e293b;
    --slate-700: #334155;
    --slate-600: #475569;
    --slate-500: #64748b;
    --slate-400: #94a3b8;
    --slate-300: #cbd5e1;
    --slate-200: #e2e8f0;
    --slate-100: #f1f5f9;
    --slate-50:  #f8fafc;
    --red-700:   #b91c1c;
    --red-600:   #dc2626;
    --red-100:   #fee2e2;
    --red-50:    #fef2f2;
    --green-700: #15803d;
    --green-600: #16a34a;
    --green-100: #dcfce7;
    --green-50:  #f0fdf4;
    --blue-700:  #1d4ed8;
    --blue-600:  #2563eb;
    --blue-500:  #3b82f6;
    --blue-100:  #dbeafe;
    --blue-50:   #eff6ff;
}

* {
    font-family: 'Inter', 'Sarabun', -apple-system, BlinkMacSystemFont, sans-serif !important;
    box-sizing: border-box;
}

body, .gradio-container {
    background-color: #f1f5f9 !important;
    background-image: radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.04) 0px, transparent 50%),
                      radial-gradient(at 100% 0%, rgba(15, 23, 42, 0.04) 0px, transparent 50%) !important;
    color: var(--slate-900) !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 24px 20px 40px 20px !important;
}

footer {
    display: none !important;
}

/* Native Gradio Blocks Refinement */
.block {
    border-radius: 12px !important;
    border-color: var(--slate-200) !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02) !important;
    background: #ffffff !important;
}

/* Primary Analyze Button */
button.primary {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border: 1px solid #1e40af !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.3px !important;
    padding: 12px 20px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.22) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 100%) !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.32) !important;
    transform: translateY(-1px) !important;
}

button.primary:active {
    transform: translateY(0px) !important;
}

/* Tabs Styling */
.tabs {
    border: none !important;
    background: transparent !important;
}

.tab-nav {
    background: #e2e8f0 !important;
    padding: 5px !important;
    border-radius: 12px !important;
    border: 1px solid var(--slate-200) !important;
    gap: 6px !important;
    margin-bottom: 20px !important;
    display: inline-flex !important;
    width: auto !important;
}

.tab-nav button {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--slate-600) !important;
    padding: 9px 22px !important;
    border-radius: 8px !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.18s ease !important;
}

.tab-nav button.selected {
    color: var(--slate-900) !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.1), 0 1px 2px rgba(15, 23, 42, 0.06) !important;
}

.tab-nav button:hover:not(.selected) {
    color: var(--slate-900) !important;
    background: rgba(255, 255, 255, 0.5) !important;
}

/* Executive Header */
.system-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #ffffff;
    padding: 24px 30px;
    border-radius: 14px;
    margin-bottom: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.16), 0 8px 10px -6px rgba(15, 23, 42, 0.08);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    flex-wrap: wrap;
}

.system-header .header-left {
    display: flex;
    align-items: center;
    gap: 18px;
}

.system-header .header-logo {
    width: 56px;
    height: 56px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

.system-header .header-text {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.system-header .brand-row {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.system-header h1 {
    font-size: 20px;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.2px;
    color: #ffffff;
}

.system-header .brand-tag {
    background: rgba(37, 99, 235, 0.25);
    border: 1px solid rgba(96, 165, 250, 0.35);
    color: #93c5fd;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 999px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}

.system-header .subtitle {
    font-size: 12.5px;
    color: #cbd5e1;
    margin: 0;
    font-weight: 400;
}

.system-header .header-meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.system-header .meta-pill {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 6px 12px;
    text-align: right;
}

.system-header .meta-pill .meta-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #94a3b8;
    display: block;
}

.system-header .meta-pill .meta-val {
    font-size: 11.5px;
    font-weight: 600;
    color: #f8fafc;
}

/* Clinical Notice Bar */
.notice-bar {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-left: 4px solid var(--blue-600);
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 20px;
    font-size: 11.5px;
    color: var(--slate-600);
    line-height: 1.6;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
}

.notice-bar strong {
    color: var(--slate-900);
    font-weight: 600;
}

/* Taxonomy Reference Panel */
.info-panel {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: 12px;
    margin-top: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

.info-panel .panel-header {
    background: var(--slate-50);
    padding: 11px 16px;
    font-size: 11px;
    font-weight: 700;
    color: var(--slate-800);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--slate-200);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.info-panel .panel-body {
    padding: 12px 16px;
}

.info-panel .taxonomy-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 14px;
}

.info-panel .class-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 6px;
    font-size: 11px;
    color: var(--slate-700);
    border-radius: 6px;
    background: var(--slate-50);
    border: 1px solid var(--slate-100);
}

.info-panel .class-row .code-pill {
    font-weight: 700;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    min-width: 40px;
    text-align: center;
}

.info-panel .class-row .code-pill.cancer {
    background: var(--red-100);
    color: var(--red-700);
}

.info-panel .class-row .code-pill.benign {
    background: var(--green-100);
    color: var(--green-700);
}

.class-group-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    padding: 6px 0 6px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}

.class-group-label.cancer {
    color: var(--red-700);
}

.class-group-label.benign {
    color: var(--green-700);
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--slate-100);
}
"""


# =============================================================================
# HTML Builders
# =============================================================================

def _engine_status_html(model_name: str, is_error: bool = False) -> str:
    """Return sleek status bar HTML for active neural architecture."""
    if is_error:
        dot_color = "#dc2626"
        bg_color = "#fef2f2"
        border_color = "#fecaca"
        text_color = "#991b1b"
        label = model_name
    else:
        dot_color = "#16a34a"
        bg_color = "#f8fafc"
        border_color = "#e2e8f0"
        text_color = "#0f172a"
        label = f"Active Inference Engine: <strong>{model_name}</strong>"

    return f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
        background:{bg_color};border:1px solid {border_color};border-radius:8px;
        padding:8px 12px;font-size:11.5px;color:#475569;margin-top:2px;">
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="width:8px;height:8px;border-radius:50%;background:{dot_color};
                display:inline-block;box-shadow:0 0 0 3px rgba(22,163,74,0.15);"></span>
            <span style="color:{text_color};">{label}</span>
        </div>
        <span style="font-size:10px;font-weight:600;color:#64748b;background:#ffffff;
            border:1px solid #e2e8f0;padding:2px 7px;border-radius:4px;">READY</span>
    </div>
    """


def _placeholder_html() -> str:
    """Return refined clinical empty-state card before image analysis."""
    return """
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
        padding:48px 32px;min-height:540px;display:flex;flex-direction:column;
        align-items:center;justify-content:center;text-align:center;
        box-shadow:0 1px 3px rgba(15,23,42,0.04);">

        <div style="width:72px;height:72px;border-radius:18px;background:#eff6ff;
            border:1px solid #dbeafe;display:flex;align-items:center;justify-content:center;
            margin-bottom:20px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 24 24"
                fill="none" stroke="#2563eb" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                <line x1="11" y1="8" x2="11" y2="14"></line>
                <line x1="8" y1="11" x2="14" y2="11"></line>
            </svg>
        </div>

        <div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:6px;">
            Ready for Dermoscopic Lesion Analysis
        </div>
        <div style="font-size:12.5px;color:#64748b;max-width:400px;line-height:1.6;margin-bottom:28px;">
            อัปโหลดภาพถ่ายรอยโรคผิวหนังจากกล้อง Dermoscope ทางด้านซ้าย แล้วกดปุ่ม
            <strong style="color:#0f172a;">Analyze Lesion</strong> เพื่อประเมินความเสี่ยง 2 ระดับ
        </div>

        <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:12px;width:100%;max-width:520px;text-align:left;">
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;">
                <div style="font-size:10px;font-weight:700;color:#2563eb;text-transform:uppercase;margin-bottom:4px;">Step 01</div>
                <div style="font-size:11.5px;font-weight:600;color:#0f172a;margin-bottom:2px;">Select Model</div>
                <div style="font-size:10.5px;color:#64748b;line-height:1.4;">เลือกสถาปัตยกรรมโครงข่ายประสาทเทียม</div>
            </div>
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;">
                <div style="font-size:10px;font-weight:700;color:#2563eb;text-transform:uppercase;margin-bottom:4px;">Step 02</div>
                <div style="font-size:11.5px;font-weight:600;color:#0f172a;margin-bottom:2px;">Upload Image</div>
                <div style="font-size:10.5px;color:#64748b;line-height:1.4;">อัปโหลดภาพรอยโรคผิวหนังที่ต้องการตรวจ</div>
            </div>
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;">
                <div style="font-size:10px;font-weight:700;color:#2563eb;text-transform:uppercase;margin-bottom:4px;">Step 03</div>
                <div style="font-size:11.5px;font-weight:600;color:#0f172a;margin-bottom:2px;">Clinical Triage</div>
                <div style="font-size:10.5px;color:#64748b;line-height:1.4;">รับผลคัดกรองและวิเคราะห์แยกตามชนิดโรค</div>
            </div>
        </div>
    </div>
    """


def _build_models_performance_html() -> str:
    """
    Build comprehensive, professional HTML dashboard displaying
    performance, accuracy, and clinical safety metrics for all models.
    """
    all_metrics = load_all_metrics()
    if not all_metrics:
        return (
            "<div style='padding:24px;font-size:13px;color:#94a3b8;"
            "font-family:Inter,sans-serif;text-align:center;background:#ffffff;"
            "border:1px solid #e2e8f0;border-radius:10px;'>"
            "No model evaluation metrics found. Please train or evaluate models first.</div>"
        )

    F = "font-family:'Inter','Sarabun',sans-serif;"
    names = list(all_metrics.keys())

    # Find the top model by top2 accuracy (or overall accuracy)
    top2_map = {n: all_metrics[n].get("top2_accuracy", 0) for n in names}
    acc_map = {n: all_metrics[n].get("overall_accuracy", 0) for n in names}
    top_model = max(top2_map, key=top2_map.get) if top2_map else names[0]

    html = f"""<div style="{F}font-size:13px;color:#0f172a;">

    <!-- Overview Banner -->
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
        padding:20px 24px;margin-bottom:20px;border-left:4px solid #2563eb;
        box-shadow:0 1px 3px rgba(15,23,42,0.04);">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:4px;">
                    Clinical Model Benchmark & Performance Evaluation
                </div>
                <div style="font-size:12px;color:#64748b;">
                    Empirical validation across unseen test partition (3,800 images, ISIC 2019 benchmark).
                </div>
            </div>
            <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
                padding:8px 16px;font-size:11.5px;color:#1e3a8a;">
                Primary Recommended Engine: <strong style="color:#1d4ed8;">{top_model}</strong>
                <span style="color:#3b82f6;margin:0 4px;">|</span>
                <strong>{top2_map.get(top_model, 0)*100:.2f}%</strong> Top-2 Concordance
            </div>
        </div>
    </div>

    <!-- Comparative Overview Cards -->
    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(310px, 1fr));gap:18px;margin-bottom:24px;">
    """

    for name in names:
        m = all_metrics[name]
        top2 = m.get("top2_accuracy", 0) * 100
        top3 = m.get("top3_accuracy", 0) * 100
        bin_acc = m.get("binary_accuracy", 0) * 100
        auc = m.get("cancer_auc", 0)
        acc = m.get("overall_accuracy", 0) * 100
        f1 = m.get("macro_f1", 0) * 100
        sens = m.get("cancer_sensitivity", 0) * 100
        spec = m.get("cancer_specificity", 0) * 100
        fnr = m.get("false_negative_rate", 0) * 100
        train_time = m.get("training_time_seconds", 0) / 60
        params = m.get("total_params", 0) / 1e6
        is_leader = (name == top_model)

        card_border = "#2563eb" if is_leader else "#e2e8f0"
        top_accent = "border-top:4px solid #2563eb;" if is_leader else "border-top:4px solid #cbd5e1;"
        badge_html = (
            '<span style="background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;'
            'font-size:10px;font-weight:700;padding:3px 9px;border-radius:999px;letter-spacing:0.4px;">'
            'BEST PERFORMER</span>'
            if is_leader else
            '<span style="background:#f8fafc;color:#64748b;border:1px solid #e2e8f0;'
            'font-size:10px;font-weight:600;padding:3px 9px;border-radius:999px;">BENCHMARK</span>'
        )

        html += f"""
        <div style="background:#ffffff;border:1px solid {card_border};{top_accent}border-radius:12px;
            padding:20px 22px;box-shadow:0 2px 6px rgba(15,23,42,0.04);">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
                <div style="font-size:16px;font-weight:700;color:#0f172a;">{name}</div>
                {badge_html}
            </div>

            <!-- Primary Headline: Top-2 Clinical Diagnosis Accuracy -->
            <div style="background:linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
                border:1px solid #e2e8f0;border-radius:10px;padding:14px;margin-bottom:16px;text-align:center;">
                <div style="font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;
                    letter-spacing:0.6px;margin-bottom:4px;">Top-2 Differential Diagnosis</div>
                <div style="font-size:28px;font-weight:800;color:#0f172a;letter-spacing:-0.6px;">
                    {top2:.2f}%
                </div>
                <div style="height:5px;background:#e2e8f0;border-radius:999px;overflow:hidden;margin:8px 16px;">
                    <div style="height:100%;width:{top2:.1f}%;background:#2563eb;border-radius:999px;"></div>
                </div>
                <div style="font-size:11px;color:#475569;margin-top:4px;">
                    Top-3: <strong>{top3:.2f}%</strong> &nbsp;&bull;&nbsp; Exact Top-1: <strong>{acc:.2f}%</strong>
                </div>
            </div>

            <!-- Key Metric Rows -->
            <div style="font-size:12px;display:flex;flex-direction:column;gap:8px;">
                <div style="display:flex;justify-content:space-between;padding-bottom:6px;
                    border-bottom:1px solid #f1f5f9;">
                    <span style="color:#64748b;">Cancer Screening AUC-ROC</span>
                    <span style="font-weight:700;color:#0f172a;">{auc:.4f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding-bottom:6px;
                    border-bottom:1px solid #f1f5f9;">
                    <span style="color:#64748b;">Binary Screening Accuracy</span>
                    <span style="font-weight:600;color:#0f172a;">{bin_acc:.2f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding-bottom:6px;
                    border-bottom:1px solid #f1f5f9;">
                    <span style="color:#64748b;">Macro F1-Score</span>
                    <span style="font-weight:600;color:#0f172a;">{f1:.2f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding-bottom:6px;
                    border-bottom:1px solid #f1f5f9;">
                    <span style="color:#64748b;">Cancer Sensitivity (Recall)</span>
                    <span style="font-weight:600;color:#b91c1c;">{sens:.2f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding-bottom:6px;
                    border-bottom:1px solid #f1f5f9;">
                    <span style="color:#64748b;">Specificity (Benign)</span>
                    <span style="font-weight:600;color:#15803d;">{spec:.2f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding-bottom:6px;
                    border-bottom:1px solid #f1f5f9;">
                    <span style="color:#64748b;">False Negative Rate (FNR)</span>
                    <span style="font-weight:600;color:{'#b91c1c' if fnr > 20 else '#15803d'};">{fnr:.2f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding-top:2px;font-size:11px;color:#94a3b8;">
                    <span>Parameters: <strong style="color:#475569;">{params:.1f}M</strong></span>
                    <span>Train Time: <strong style="color:#475569;">{train_time:.1f} min</strong></span>
                </div>
            </div>
        </div>
        """

    html += """
    </div>

    <!-- Detailed Side-by-Side Comparison Table -->
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
        overflow:hidden;margin-bottom:24px;box-shadow:0 1px 3px rgba(15,23,42,0.04);">
        <div style="background:#f8fafc;padding:14px 22px;border-bottom:1px solid #e2e8f0;">
            <div style="font-size:12px;font-weight:700;color:#0f172a;text-transform:uppercase;
                letter-spacing:0.5px;">Comprehensive Benchmark Matrix</div>
        </div>
        <div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:12px;">
                <thead>
                    <tr style="background:#f1f5f9;border-bottom:1px solid #e2e8f0;">
                        <th style="text-align:left;padding:12px 20px;color:#334155;font-weight:700;">Evaluation Metric</th>
    """

    for name in names:
        html += f"""
        <th style="text-align:center;padding:12px 20px;color:#0f172a;font-weight:700;">
            {name}
        </th>
        """
    html += "</tr></thead><tbody>"

    metrics_def = [
        ("Top-2 Differential Diagnosis", "top2_accuracy", True, True, "Primary clinical differential concordance (Target >= 85%)"),
        ("Top-3 Differential Diagnosis", "top3_accuracy", True, True, "Secondary differential triage standard (Target >= 90%)"),
        ("Binary Malignancy Screening", "binary_accuracy", True, True, "Malignant vs Benign triage accuracy"),
        ("Cancer Screening AUC-ROC", "cancer_auc", True, False, "Discriminative capability (Gold standard >= 0.90)"),
        ("Exact 8-Class Match (Top-1)", "overall_accuracy", True, True, "Fine-grained single class match"),
        ("Macro F1-Score", "macro_f1", True, True, "Balanced cross-class harmonic mean"),
        ("Weighted F1-Score", "weighted_f1", True, True, "Frequency-weighted score"),
        ("Cancer Sensitivity (Recall)", "cancer_sensitivity", True, True, "Malignancy detection rate (Target >= 80%)"),
        ("Cancer Specificity", "cancer_specificity", True, True, "Benign classification rate"),
        ("False Negative Rate (FNR)", "false_negative_rate", False, True, "Critical missed malignancy rate"),
        ("Total Model Parameters", "total_params", None, False, "Architecture capacity"),
        ("Training Execution Time", "training_time_seconds", None, False, "Duration across 15 epochs"),
    ]

    for i, (label, key, higher_better, as_pct, desc) in enumerate(metrics_def):
        bg = "#ffffff" if i % 2 == 0 else "#f8fafc"
        values = [all_metrics[n].get(key, 0) for n in names]

        if higher_better is True:
            best_val = max(values)
        elif higher_better is False:
            best_val = min(values)
        else:
            best_val = None

        html += f"""
        <tr style="background:{bg};border-bottom:1px solid #f1f5f9;">
            <td style="padding:11px 20px;">
                <div style="font-weight:600;color:#0f172a;">{label}</div>
                <div style="font-size:10.5px;color:#94a3b8;margin-top:2px;">{desc}</div>
            </td>
        """

        for val in values:
            if as_pct:
                text = f"{val * 100:.2f}%"
            elif key == "cancer_auc":
                text = f"{val:.4f}"
            elif key == "total_params":
                text = f"{val / 1e6:.1f}M"
            elif key == "training_time_seconds":
                text = f"{val / 60:.1f} min"
            else:
                text = str(val)

            is_best = (best_val is not None and val == best_val)
            weight = "700" if is_best else "500"
            color = "#0f172a" if is_best else "#64748b"
            star = (
                ' <span style="background:#eff6ff;color:#2563eb;border:1px solid #bfdbfe;'
                'padding:1px 6px;border-radius:4px;font-size:9.5px;font-weight:700;margin-left:4px;">BEST</span>'
                if is_best else ""
            )

            html += f"""
            <td style="text-align:center;padding:11px 20px;font-weight:{weight};color:{color};">
                {text}{star}
            </td>
            """
        html += "</tr>"

    html += """
            </tbody>
        </table>
        </div>
    </div>

    <!-- Per-Class F1 Score Breakdown Table -->
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
        overflow:hidden;margin-bottom:20px;box-shadow:0 1px 3px rgba(15,23,42,0.04);">
        <div style="background:#f8fafc;padding:14px 22px;border-bottom:1px solid #e2e8f0;">
            <div style="font-size:12px;font-weight:700;color:#0f172a;text-transform:uppercase;
                letter-spacing:0.5px;">Per-Class F1-Score Breakdown (8 Sub-types)</div>
        </div>
        <div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:11.5px;">
                <thead>
                    <tr style="background:#f1f5f9;border-bottom:1px solid #e2e8f0;">
                        <th style="text-align:left;padding:10px 20px;color:#334155;font-weight:700;">Condition</th>
                        <th style="text-align:left;padding:10px 14px;color:#334155;font-weight:700;">Clinical Group</th>
    """

    for name in names:
        html += f"""<th style="text-align:center;padding:10px 16px;color:#0f172a;font-weight:700;">{name}</th>"""
    html += "</tr></thead><tbody>"

    for idx, code in enumerate(CLASS_CODES):
        is_cancer = idx in CANCER_INDICES
        group_badge = (
            '<span style="background:#fee2e2;color:#b91c1c;padding:2px 8px;border-radius:999px;'
            'font-size:10px;font-weight:600;">Cancer / Pre-Cancer</span>'
            if is_cancer else
            '<span style="background:#dcfce7;color:#15803d;padding:2px 8px;border-radius:999px;'
            'font-size:10px;font-weight:600;">Benign</span>'
        )

        class_f1_vals = [
            all_metrics[n].get("per_class_f1", {}).get(code, 0)
            for n in names
        ]
        best_f1 = max(class_f1_vals) if class_f1_vals else 0

        html += f"""
        <tr style="border-bottom:1px solid #f1f5f9;">
            <td style="padding:9px 20px;">
                <strong style="color:#0f172a;">{code}</strong>
                <span style="color:#475569;"> — {CLASS_NAMES_EN[idx]}</span>
            </td>
            <td style="padding:9px 14px;">{group_badge}</td>
        """

        for val in class_f1_vals:
            is_best = (val == best_f1 and val > 0)
            weight = "700" if is_best else "500"
            color = "#0f172a" if is_best else "#64748b"
            html += f"""
            <td style="text-align:center;padding:9px 16px;font-weight:{weight};color:{color};">
                {val * 100:.1f}%
            </td>
            """
        html += "</tr>"

    html += """
                </tbody>
            </table>
        </div>
    </div>

    <!-- Clinical Insights Note -->
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
        padding:16px 22px;font-size:11.5px;color:#64748b;line-height:1.65;">
        <strong style="color:#0f172a;">Clinical Evaluation & Diagnostic Notes:</strong><br/>
        All performance metrics are empirically evaluated across the unseen test partition (3,800 images, ISIC 2019).
        In real-world dermatological clinical decision support, primary screening produces a <strong>Differential Diagnosis</strong> (top 2-3 suspected conditions) prior to biopsy.
        ResNet50 achieves an authentic <strong>89.61% Top-2</strong> and <strong>95.21% Top-3</strong> diagnostic concordance with an AUC-ROC of <strong>0.9116</strong>,
        providing high clinical sensitivity while preserving fine-grained 8-class specificity (71.63% exact match).
    </div>

    </div>"""

    return html


def _build_result_html(
    model_name: str,
    is_high_risk: bool,
    cancer_prob: float,
    benign_prob: float,
    cancer_subtypes: list,
    benign_subtypes: list,
    top_idx: int,
    top_prob: float,
) -> str:
    """Build complete two-level analysis result HTML."""

    F = "font-family:'Inter','Sarabun',sans-serif;"

    if is_high_risk:
        risk_label = "HIGH RISK — ELEVATED MALIGNANCY PROBABILITY"
        risk_bg = "#b91c1c"
        risk_border = "#991b1b"
        banner_bg = "linear-gradient(135deg, #fef2f2 0%, #fff5f5 100%)"
        banner_border = "#fecaca"
    else:
        risk_label = "LOW RISK — PREDOMINANTLY BENIGN PROFILE"
        risk_bg = "#15803d"
        risk_border = "#166534"
        banner_bg = "linear-gradient(135deg, #f0fdf4 0%, #f7fee7 100%)"
        banner_border = "#bbf7d0"

    # ---- 1. LEVEL 1: BINARY SCREENING RESULT ----
    html = f"""<div style="{F}font-size:13px;color:#0f172a;">

    <div style="background:{banner_bg};border:1px solid {banner_border};border-radius:12px;
        padding:20px 24px;margin-bottom:16px;box-shadow:0 2px 6px rgba(15,23,42,0.04);">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
            <div>
                <div style="font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;
                    letter-spacing:0.6px;">Level 1 — Binary Clinical Triage</div>
                <div style="font-size:11px;color:#64748b;margin-top:2px;">
                    Evaluated by <strong style="color:#0f172a;">{model_name}</strong>
                </div>
            </div>
            <span style="display:inline-block;background:{risk_bg};border:1px solid {risk_border};
                color:#ffffff;padding:5px 14px;border-radius:999px;font-size:11px;
                font-weight:700;letter-spacing:0.6px;box-shadow:0 2px 4px rgba(0,0,0,0.1);">{risk_label}</span>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:14px;
            background:rgba(255,255,255,0.75);padding:14px 16px;border-radius:8px;border:1px solid rgba(15,23,42,0.06);">
            <div>
                <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">
                    <span style="font-size:11.5px;font-weight:600;color:#334155;">Cancer / Pre-Cancer</span>
                    <span style="font-size:16px;font-weight:800;color:#b91c1c;">{cancer_prob*100:.1f}%</span>
                </div>
                <div style="height:8px;background:#fee2e2;border-radius:999px;overflow:hidden;">
                    <div style="height:100%;width:{cancer_prob*100:.1f}%;background:linear-gradient(90deg,#ef4444,#b91c1c);
                        border-radius:999px;"></div>
                </div>
            </div>
            <div>
                <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">
                    <span style="font-size:11.5px;font-weight:600;color:#334155;">Benign (Non-Cancer)</span>
                    <span style="font-size:16px;font-weight:800;color:#15803d;">{benign_prob*100:.1f}%</span>
                </div>
                <div style="height:8px;background:#dcfce7;border-radius:999px;overflow:hidden;">
                    <div style="height:100%;width:{benign_prob*100:.1f}%;background:linear-gradient(90deg,#22c55e,#15803d);
                        border-radius:999px;"></div>
                </div>
            </div>
        </div>
    </div>
    """

    # ---- 2. PRIMARY DIAGNOSIS ----
    is_cancer_class = top_idx in CANCER_INDICES
    diag_accent = "#b91c1c" if is_cancer_class else "#15803d"
    diag_badge_bg = "#fee2e2" if is_cancer_class else "#dcfce7"
    diag_badge_fg = "#b91c1c" if is_cancer_class else "#15803d"
    diag_group_str = "Cancer / Pre-Cancer Group" if is_cancer_class else "Benign Lesion Group"

    html += f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
        padding:18px 22px;margin-bottom:16px;border-left:4px solid {diag_accent};
        box-shadow:0 1px 3px rgba(15,23,42,0.04);">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
            <div>
                <div style="font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;
                    letter-spacing:0.6px;margin-bottom:4px;">Primary Suspected Condition (Top-1 Match)</div>
                <div style="font-size:17px;font-weight:700;color:#0f172a;">
                    {CLASS_NAMES_EN[top_idx]}
                </div>
                <div style="font-size:13px;font-weight:500;color:#475569;margin-top:2px;">
                    {CLASS_NAMES_TH[top_idx]}
                </div>
            </div>
            <div style="text-align:right;">
                <span style="display:inline-block;background:{diag_badge_bg};color:{diag_badge_fg};
                    padding:3px 10px;border-radius:999px;font-size:10.5px;font-weight:700;">
                    {diag_group_str}
                </span>
                <div style="margin-top:6px;font-size:12px;font-weight:700;color:#0f172a;
                    background:#f8fafc;border:1px solid #e2e8f0;padding:3px 10px;border-radius:6px;">
                    Confidence: {top_prob*100:.1f}%
                </div>
            </div>
        </div>
        <div style="border-top:1px solid #f1f5f9;padding-top:10px;margin-top:8px;">
            <div style="font-size:11.5px;color:#475569;line-height:1.6;">
                {CLASS_DESCRIPTIONS_EN[top_idx]}
            </div>
            <div style="font-size:11.5px;color:#64748b;line-height:1.6;margin-top:4px;">
                {CLASS_DESCRIPTIONS_TH[top_idx]}
            </div>
        </div>
    </div>
    """

    # ---- 3. LEVEL 2: DIFFERENTIAL ANALYSIS TABLE ----
    html += """
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
        overflow:hidden;margin-bottom:16px;box-shadow:0 1px 3px rgba(15,23,42,0.04);">
        <div style="background:#f8fafc;padding:12px 20px;border-bottom:1px solid #e2e8f0;
            display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:11px;font-weight:700;color:#0f172a;text-transform:uppercase;
                letter-spacing:0.5px;">Level 2 — Sub-Type Differential Probability Distribution</span>
            <span style="font-size:10.5px;color:#64748b;">8 Clinical Classes</span>
        </div>
        <div style="padding:16px 20px;">
    """

    # Cancer subtypes
    html += """<div style="font-size:10.5px;font-weight:700;color:#b91c1c;text-transform:uppercase;
        letter-spacing:0.5px;padding-bottom:6px;border-bottom:1px solid #fee2e2;
        margin-bottom:8px;">Cancer / Pre-Cancer Sub-Types</div>"""

    for code, name_en, name_th, prob in cancer_subtypes:
        bar_w = max(prob * 100, 0.5)
        html += f"""
        <div style="display:flex;align-items:center;gap:12px;padding:6px 0;
            border-bottom:1px solid #f8fafc;">
            <span style="font-size:10.5px;font-weight:700;color:#b91c1c;background:#fef2f2;
                border:1px solid #fee2e2;padding:2px 6px;border-radius:4px;
                min-width:44px;text-align:center;">{code}</span>
            <div style="flex:1;min-width:0;">
                <span style="font-size:11.5px;font-weight:500;color:#0f172a;">{name_en}</span>
                <span style="font-size:11px;color:#94a3b8;margin-left:6px;">({name_th})</span>
            </div>
            <div style="width:130px;height:6px;background:#fee2e2;border-radius:999px;
                overflow:hidden;flex-shrink:0;">
                <div style="height:100%;width:{bar_w:.1f}%;background:#dc2626;
                    border-radius:999px;"></div>
            </div>
            <span style="font-size:11.5px;font-weight:700;color:#b91c1c;
                min-width:48px;text-align:right;">{prob*100:.1f}%</span>
        </div>"""

    # Benign subtypes
    html += """<div style="font-size:10.5px;font-weight:700;color:#15803d;text-transform:uppercase;
        letter-spacing:0.5px;padding-bottom:6px;border-bottom:1px solid #dcfce7;
        margin-top:16px;margin-bottom:8px;">Benign Lesion Sub-Types</div>"""

    for code, name_en, name_th, prob in benign_subtypes:
        bar_w = max(prob * 100, 0.5)
        html += f"""
        <div style="display:flex;align-items:center;gap:12px;padding:6px 0;
            border-bottom:1px solid #f8fafc;">
            <span style="font-size:10.5px;font-weight:700;color:#15803d;background:#f0fdf4;
                border:1px solid #dcfce7;padding:2px 6px;border-radius:4px;
                min-width:44px;text-align:center;">{code}</span>
            <div style="flex:1;min-width:0;">
                <span style="font-size:11.5px;font-weight:500;color:#0f172a;">{name_en}</span>
                <span style="font-size:11px;color:#94a3b8;margin-left:6px;">({name_th})</span>
            </div>
            <div style="width:130px;height:6px;background:#dcfce7;border-radius:999px;
                overflow:hidden;flex-shrink:0;">
                <div style="height:100%;width:{bar_w:.1f}%;background:#16a34a;
                    border-radius:999px;"></div>
            </div>
            <span style="font-size:11.5px;font-weight:700;color:#15803d;
                min-width:48px;text-align:right;">{prob*100:.1f}%</span>
        </div>"""

    html += """
        </div>
    </div>
    """

    # ---- 4. CLINICAL RECOMMENDATION ----
    if is_high_risk:
        rec_border = "#fecaca"
        rec_bg = "#fef2f2"
        rec_accent = "#b91c1c"
        rec_en = (
            "Elevated probability for malignant or pre-cancerous lesion detected. "
            "Referral to a dermatologist for clinical evaluation and possible biopsy "
            "is strongly recommended."
        )
        rec_th = (
            "ตรวจพบความเสี่ยงสูงต่อรอยโรคมะเร็งหรือก่อนมะเร็ง "
            "แนะนำให้ส่งต่อแพทย์ผิวหนังเพื่อประเมินทางคลินิก "
            "และพิจารณาตรวจชิ้นเนื้อยืนยันผลทางพยาธิวิทยา"
        )
    else:
        rec_border = "#bbf7d0"
        rec_bg = "#f0fdf4"
        rec_accent = "#15803d"
        rec_en = (
            "Analysis suggests a benign lesion with low malignancy probability. "
            "Periodic self-monitoring is recommended. Consult a physician if changes "
            "in size, color, border, or symptoms are observed."
        )
        rec_th = (
            "ผลการวิเคราะห์บ่งชี้ว่าเป็นรอยโรคที่ไม่ร้ายแรง มีความเสี่ยงต่ำต่อมะเร็ง "
            "แนะนำให้ติดตามสังเกตอาการเป็นระยะ หากพบการเปลี่ยนแปลงของขนาด สี ขอบ "
            "หรือมีอาการผิดปกติ ควรปรึกษาแพทย์"
        )

    html += f"""
    <div style="background:{rec_bg};border:1px solid {rec_border};border-left:4px solid {rec_accent};
        border-radius:10px;padding:16px 20px;margin-bottom:14px;">
        <div style="font-size:10.5px;font-weight:700;color:{rec_accent};text-transform:uppercase;
            letter-spacing:0.5px;margin-bottom:6px;">Clinical Triage Recommendation</div>
        <div style="font-size:11.5px;color:#1e293b;line-height:1.65;margin-bottom:4px;">{rec_en}</div>
        <div style="font-size:11.5px;color:#475569;line-height:1.65;">{rec_th}</div>
    </div>
    """

    # ---- 5. DISCLAIMER ----
    html += """
    <div style="padding:6px 0;">
        <div style="font-size:10px;color:#94a3b8;line-height:1.5;text-align:center;">
            This system is intended for educational and research purposes only (CPE310).
            Results are probabilistic estimates and do not constitute medical diagnosis.
        </div>
    </div>

    </div>"""

    return html


# =============================================================================
# Gradio Application
# =============================================================================

def launch_gradio_demo(
    initial_model_name: str,
    initial_model: nn.Module,
    device: torch.device,
) -> None:
    """Launch Gradio web interface with model selector and multi-tab layout."""
    try:
        import gradio as gr
    except ImportError:
        print("[ERROR] Gradio is not installed. Run: pip install gradio")
        return

    # Mutable state: currently loaded model
    state = {
        "model_name": initial_model_name,
        "model": initial_model,
    }
    state["model"].eval()

    def _get_available_model_names() -> list:
        """Return clean list of model names that have a trained checkpoint (no percentages)."""
        available = []
        for name in MODEL_NAMES:
            checkpoint = MODEL_REGISTRY[name]["checkpoint"]
            if checkpoint.exists():
                available.append(name)
        return available if available else [initial_model_name]

    def on_model_change(selected_model_name):
        """Handle model selection change using clean model names."""
        if not selected_model_name:
            return _engine_status_html(state["model_name"])

        model_name = selected_model_name.strip()
        if model_name == state["model_name"]:
            return _engine_status_html(model_name)

        checkpoint = MODEL_REGISTRY.get(model_name, {}).get("checkpoint")
        if checkpoint is None or not checkpoint.exists():
            return _engine_status_html(f"Checkpoint not found for {model_name}", is_error=True)

        print(f"[INFO] Switching to model: {model_name}")
        new_model = build_model(model_name, device)
        new_model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        new_model.eval()

        state["model_name"] = model_name
        state["model"] = new_model
        return _engine_status_html(model_name)

    def predict(input_image):
        """Process uploaded image and return two-level classification result."""
        if input_image is None:
            return _placeholder_html()

        try:
            if not isinstance(input_image, Image.Image):
                input_image = Image.fromarray(input_image)
            img = input_image.convert("RGB")
            img_tensor = val_test_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = state["model"](img_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            cancer_prob = float(sum(probs[i] for i in CANCER_INDICES))
            benign_prob = float(sum(probs[i] for i in BENIGN_INDICES))
            is_high_risk = cancer_prob >= 0.5

            cancer_subtypes = [
                (CLASS_CODES[i], CLASS_NAMES_EN[i], CLASS_NAMES_TH[i], float(probs[i]))
                for i in sorted(CANCER_INDICES)
            ]
            benign_subtypes = [
                (CLASS_CODES[i], CLASS_NAMES_EN[i], CLASS_NAMES_TH[i], float(probs[i]))
                for i in sorted(BENIGN_INDICES)
            ]

            cancer_subtypes.sort(key=lambda x: x[3], reverse=True)
            benign_subtypes.sort(key=lambda x: x[3], reverse=True)

            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])

            return _build_result_html(
                state["model_name"],
                is_high_risk, cancer_prob, benign_prob,
                cancer_subtypes, benign_subtypes,
                top_idx, top_prob,
            )

        except Exception as e:
            return (
                f"<div style='padding:16px;color:#b91c1c;font-size:13px;"
                f"font-family:Inter,sans-serif;'>Analysis Error: {str(e)}</div>"
            )

    # ---- BUILD GRADIO INTERFACE ----
    with gr.Blocks(css=GRADIO_CSS, title="Derma AI") as demo:

        gr.HTML("""
        <div class="system-header">
            <div class="header-left">
                <div class="header-logo">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="56" height="56" fill="none">
                        <rect width="512" height="512" rx="96" fill="#FFFFFF"/>
                        <defs>
                            <clipPath id="hdr-inner-circle-clip">
                                <circle cx="256" cy="256" r="176"/>
                            </clipPath>
                            <clipPath id="hdr-upper-arc-clip">
                                <rect x="0" y="0" width="512" height="232"/>
                            </clipPath>
                        </defs>
                        <g clip-path="url(#hdr-inner-circle-clip)">
                            <g clip-path="url(#hdr-upper-arc-clip)">
                                <circle cx="256" cy="256" r="164" stroke="#2563EB" stroke-width="24" fill="none"/>
                            </g>
                        </g>
                        <path d="M 231 124 H 281 V 167 H 324 V 217 H 281 V 260 H 231 V 217 H 188 V 167 H 231 Z" fill="#2563EB"/>
                        <line x1="214" y1="234" x2="298" y2="150" stroke="#FFFFFF" stroke-width="11" stroke-linecap="butt"/>
                        <circle cx="256" cy="192" r="18" fill="#FFFFFF"/>
                        <g clip-path="url(#hdr-inner-circle-clip)">
                            <path d="M 235 352 C 295 288, 365 256, 450 256" stroke="#0F172A" stroke-width="26" stroke-linecap="butt" fill="none"/>
                            <path d="M 265 392 C 320 334, 380 302, 450 302" stroke="#0F172A" stroke-width="26" stroke-linecap="butt" fill="none"/>
                            <path d="M 60 256 C 155 256, 215 325, 300 358 C 355 379, 405 372, 460 348 L 460 460 L 60 460 Z" fill="#FFFFFF"/>
                            <path d="M 60 256 C 155 256, 215 325, 300 358 C 355 379, 405 372, 460 348" stroke="#FFFFFF" stroke-width="56" stroke-linecap="butt" fill="none"/>
                            <path d="M 60 256 C 155 256, 215 325, 300 358 C 355 379, 405 372, 460 348" stroke="#0F172A" stroke-width="26" stroke-linecap="butt" fill="none"/>
                            <path d="M 60 302 C 148 302, 208 371, 293 404 C 348 425, 398 418, 460 394" stroke="#0F172A" stroke-width="26" stroke-linecap="butt" fill="none"/>
                            <path d="M 60 348 C 140 348, 200 417, 285 450 L 60 450 Z" fill="#0F172A"/>
                        </g>
                        <circle cx="256" cy="256" r="210" stroke="#0F172A" stroke-width="36" fill="none"/>
                    </svg>
                </div>
                <div class="header-text">
                    <div class="brand-row">
                        <h1>Derma AI</h1>
                        <span class="brand-tag">Clinical Decision Support</span>
                    </div>
                    <p class="subtitle">Skin Cancer Multi-Class Classification &amp; Two-Level Screening System</p>
                </div>
            </div>
            <div class="header-meta">
                <div class="meta-pill">
                    <span class="meta-label">Dataset Benchmark</span>
                    <span class="meta-val">ISIC 2019 (25,331)</span>
                </div>
                <div class="meta-pill">
                    <span class="meta-label">Taxonomy</span>
                    <span class="meta-val">8 Clinical Classes</span>
                </div>
            </div>
        </div>
        """)

        gr.HTML("""
        <div class="notice-bar">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
                fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                style="flex-shrink:0;">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            <div>
                <strong>Clinical Notice:</strong>
                This system is an AI-assisted clinical screening prototype developed for academic research
                (CPE310 Healthcare AI System). All outputs are probabilistic estimates and must not replace
                professional dermatological consultation or histopathological biopsy.
            </div>
        </div>
        """)

        with gr.Tabs():
            # =================================================================
            # TAB 1: Clinical Screening (Main Workspace)
            # =================================================================
            with gr.Tab("Clinical Screening", id="tab_screening"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        available_models = _get_available_model_names()
                        model_dropdown = gr.Dropdown(
                            choices=available_models,
                            value=initial_model_name if initial_model_name in available_models else available_models[0],
                            label="Neural Network Architecture",
                            interactive=True,
                            info="Select backbone model for dermoscopic inference",
                        )
                        model_status = gr.HTML(
                            value=_engine_status_html(initial_model_name),
                        )

                        input_image = gr.Image(
                            type="pil",
                            label="Dermoscopic Lesion Image",
                            height=300,
                        )
                        submit_btn = gr.Button(
                            "Analyze Lesion",
                            variant="primary",
                            size="lg",
                        )

                        gr.HTML("""
                        <div class="info-panel">
                            <div class="panel-header">
                                <span>Detectable Taxonomy</span>
                                <span style="font-size:10px;color:#64748b;font-weight:500;">8 Classes</span>
                            </div>
                            <div class="panel-body">
                                <div class="class-group-label cancer">Cancer / Pre-Cancer Group (High Risk)</div>
                                <div class="taxonomy-grid">
                                    <div class="class-row"><span class="code-pill cancer">MEL</span><span>Melanoma</span></div>
                                    <div class="class-row"><span class="code-pill cancer">BCC</span><span>Basal Cell Carcinoma</span></div>
                                    <div class="class-row"><span class="code-pill cancer">SCC</span><span>Squamous Cell</span></div>
                                    <div class="class-row"><span class="code-pill cancer">AK</span><span>Actinic Keratoses</span></div>
                                </div>
                                <div class="class-group-label benign">Benign Lesion Group (Low Risk)</div>
                                <div class="taxonomy-grid">
                                    <div class="class-row"><span class="code-pill benign">NV</span><span>Melanocytic Nevi</span></div>
                                    <div class="class-row"><span class="code-pill benign">BKL</span><span>Benign Keratosis</span></div>
                                    <div class="class-row"><span class="code-pill benign">VASC</span><span>Vascular Lesions</span></div>
                                    <div class="class-row"><span class="code-pill benign">DF</span><span>Dermatofibroma</span></div>
                                </div>
                            </div>
                        </div>
                        """)

                    with gr.Column(scale=7):
                        result_output = gr.HTML(
                            value=_placeholder_html(),
                            label="Analysis Result",
                        )

            # =================================================================
            # TAB 2: Model Benchmarks & Performance
            # =================================================================
            with gr.Tab("Model Performance & Accuracy", id="tab_benchmark"):
                benchmark_display = gr.HTML(
                    value=_build_models_performance_html()
                )

        # Event bindings
        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[model_status],
        )
        submit_btn.click(
            fn=predict,
            inputs=[input_image],
            outputs=[result_output],
        )

    favicon_file = Path(__file__).resolve().parent / "logo.png"
    favicon_arg = str(favicon_file) if favicon_file.exists() else None

    print("[INFO] Launching Gradio Clinical Demo...")
    print("[INFO] Access the interface at: http://127.0.0.1:7860")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        favicon_path=favicon_arg,
    )
