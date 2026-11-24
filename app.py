import streamlit as st
import os
import json
import pandas as pd
import joblib
import firebase_admin
import altair as alt

from firebase_admin import credentials, firestore
from pathlib import Path
from dotenv import load_dotenv


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "random_forest_appliance.pkl"  # legacy fallback
DATASET_PATH = BASE_DIR / "results" / "ml_ready_dataset.csv"
FIREBASE_KEY_PATH = BASE_DIR / "serviceAccountKey.json"

EXPECTED_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "SET_YOUR_FIREBASE_PROJECT_ID")
COLLECTION_NAME = "readings"

TIMEZONE = "Asia/Kolkata"

FEATURES = [
    "voltage_V",
    "current_A",
    "power_W",
    "frequency_Hz",
    "power_factor"
]

st.set_page_config(
    page_title="AI Smart Electricity Monitoring | NextGen IoT Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. HTML RENDERER HELPER (Prevents Code-Block Escaping)
# =====================================================

def render_html(html_str):
    """
    Renders HTML without triggering Markdown's 4-space indented code-block parser.
    Strips leading indentation from each line so that Streamlit never wraps
    it inside a white <pre><code> box.
    """
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# =====================================================
# 3. ADVANCED CYBER-ENERGY DESIGN SYSTEM (CSS)
# =====================================================

render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-dark: #070b14;
    --card-bg: rgba(15, 23, 42, 0.75);
    --cyan-accent: #06b6d4;
    --emerald-accent: #10b981;
    --amber-accent: #f59e0b;
    --violet-accent: #8b5cf6;
    --border-glass: rgba(56, 189, 248, 0.16);
}

/* Global Dark Canvas */
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], .main {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #070b14 !important;
    background: radial-gradient(ellipse 90% 60% at 50% -15%, rgba(14, 165, 233, 0.16), transparent),
                radial-gradient(ellipse 70% 50% at 90% 85%, rgba(99, 102, 241, 0.10), transparent),
                radial-gradient(circle at 10% 80%, rgba(16, 185, 129, 0.06), transparent),
                #070b14 !important;
    color: #f1f5f9 !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1440px !important;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090e1a 0%, #060911 100%) !important;
    background-color: #090e1a !important;
    border-right: 1px solid rgba(56, 189, 248, 0.14) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.6);
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
    font-size: 0.88rem;
    font-weight: 500;
}

/* Titles and Headings */
h1, h2, h3, h4, h5 {
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.02em;
    color: #f8fafc !important;
}

/* Custom Dividers */
hr {
    border-color: rgba(56, 189, 248, 0.15) !important;
    margin: 1.5rem 0 !important;
}

/* Force Vega-Lite / Altair Charts to be 100% Transparent (No white box) */
.stVegaLiteChart, [data-testid="stVegaLiteChart"], .vega-embed, .vega-embed canvas, .vega-embed svg {
    background: transparent !important;
    background-color: transparent !important;
}

/* Styled Streamlit Alert Banners */
[data-testid="stAlert"], .stAlert {
    background: rgba(15, 23, 42, 0.85) !important;
    background-color: rgba(15, 23, 42, 0.85) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(12px) !important;
    color: #e2e8f0 !important;
}

[data-testid="stAlert"] p {
    color: #e2e8f0 !important;
}

/* Primary Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: 0 4px 16px rgba(2, 132, 199, 0.35) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(2, 132, 199, 0.55) !important;
    border-color: rgba(56, 189, 248, 0.6) !important;
}

/* Inputs & Select Boxes */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background-color: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
}

div[data-baseweb="input"] input {
    color: #f8fafc !important;
    background-color: transparent !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(15, 23, 42, 0.7) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}

/* ----------------------------------------------------
   HERO BANNER COMPONENT
   ---------------------------------------------------- */
.hero-container {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(30, 41, 59, 0.75) 50%, rgba(15, 23, 42, 0.92) 100%);
    border: 1px solid rgba(56, 189, 248, 0.22);
    border-radius: 20px;
    padding: 26px 30px;
    margin-bottom: 24px;
    box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #06b6d4, #3b82f6, #8b5cf6, #10b981);
}

.hero-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 14px;
}

.pill-item {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 20px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.pill-online {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.pill-tech {
    background: rgba(6, 182, 212, 0.12);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.28);
}

.pill-grid {
    background: rgba(139, 92, 246, 0.14);
    color: #c084fc;
    border: 1px solid rgba(139, 92, 246, 0.3);
}

.hero-title-wrap {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 8px;
}

.hero-logo-box {
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.25), rgba(59, 130, 246, 0.3));
    border: 1px solid rgba(56, 189, 248, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.3);
}

.hero-title-text {
    font-family: 'Outfit', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 0%, #bae6fd 60%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 4px 0 16px 0;
    font-weight: 400;
}

.hero-arch-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}

.arch-chip {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 3px 9px;
    font-size: 0.78rem;
    color: #cbd5e1;
    font-family: 'JetBrains Mono', monospace;
}

.arch-chip strong {
    color: #38bdf8;
}

/* ----------------------------------------------------
   LIVE TELEMETRY STATUS BAR
   ---------------------------------------------------- */
.telemetry-status-bar {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.12) 0%, rgba(15, 23, 42, 0.75) 100%);
    border: 1px solid rgba(16, 185, 129, 0.28);
    border-radius: 14px;
    padding: 14px 20px;
    margin-bottom: 20px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}

.status-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.pulse-radar-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: radar-pulse 2s infinite;
}

@keyframes radar-pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.8); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.status-title-text {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 0.96rem;
    color: #34d399;
    letter-spacing: 0.02em;
}

.status-desc-text {
    font-size: 0.82rem;
    color: #94a3b8;
}

.status-chips-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}

.status-chip {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.76rem;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
}

.status-chip strong {
    color: #f1f5f9;
}

/* ----------------------------------------------------
   TELEMETRY SENSOR METRICS CARDS
   ---------------------------------------------------- */
.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 16px;
}

@media (max-width: 1100px) {
    .telemetry-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 600px) {
    .telemetry-grid { grid-template-columns: 1fr; }
}

.telemetry-grid-secondary {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

@media (max-width: 600px) {
    .telemetry-grid-secondary { grid-template-columns: 1fr; }
}

.sensor-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.9), rgba(15, 23, 42, 0.8));
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(14px);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.sensor-card:hover {
    transform: translateY(-3px);
    border-color: rgba(56, 189, 248, 0.45);
    box-shadow: 0 16px 32px -6px rgba(6, 182, 212, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.12);
}

.card-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
}

.card-icon-box {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.icon-cyan-box { background: rgba(6, 182, 212, 0.18); border: 1px solid rgba(6, 182, 212, 0.35); }
.icon-emerald-box { background: rgba(16, 185, 129, 0.18); border: 1px solid rgba(16, 185, 129, 0.35); }
.icon-amber-box { background: rgba(245, 158, 11, 0.18); border: 1px solid rgba(245, 158, 11, 0.35); }
.icon-violet-box { background: rgba(139, 92, 246, 0.18); border: 1px solid rgba(139, 92, 246, 0.35); }
.icon-teal-box { background: rgba(20, 184, 166, 0.18); border: 1px solid rgba(20, 184, 166, 0.35); }
.icon-blue-box { background: rgba(59, 130, 246, 0.18); border: 1px solid rgba(59, 130, 246, 0.35); }

.card-tag {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 3px 8px;
    border-radius: 6px;
}

.tag-green { background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
.tag-cyan { background: rgba(6, 182, 212, 0.16); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.3); }
.tag-amber { background: rgba(245, 158, 11, 0.16); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
.tag-violet { background: rgba(139, 92, 246, 0.16); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.3); }

.card-label-text {
    font-size: 0.82rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.card-value-display {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.85rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
    margin-bottom: 10px;
}

.card-unit-text {
    font-size: 1.05rem;
    font-weight: 500;
    color: #38bdf8;
    margin-left: 2px;
}

.card-footer-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    padding-top: 10px;
    font-size: 0.74rem;
    color: #64748b;
}

.card-footer-info strong {
    color: #cbd5e1;
}

/* ----------------------------------------------------
   AI APPLIANCE IDENTIFICATION SHOWCASE
   ---------------------------------------------------- */
.ai-showcase-panel {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.94) 0%, rgba(30, 41, 59, 0.85) 100%);
    border: 1px solid rgba(129, 140, 248, 0.32);
    border-radius: 20px;
    padding: 26px 28px;
    margin: 20px 0;
    box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    position: relative;
    overflow: hidden;
}

.ai-showcase-panel::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #8b5cf6, #ec4899, #3b82f6);
}

.ai-header-flex {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 22px;
}

.ai-title-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
}

.ai-chip-pill {
    background: rgba(139, 92, 246, 0.18);
    border: 1px solid rgba(139, 92, 246, 0.4);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #c084fc;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.ai-model-tag-pill {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.8rem;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
}

.ai-model-tag-pill strong {
    color: #38bdf8;
}

.ai-content-grid {
    display: grid;
    grid-template-columns: 1.2fr 1fr;
    gap: 24px;
    align-items: center;
}

@media (max-width: 900px) {
    .ai-content-grid { grid-template-columns: 1fr; }
}

.appliance-identity-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 20px;
}

.appliance-icon-ring {
    width: 68px;
    height: 68px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(139, 92, 246, 0.25));
    border: 2px solid rgba(56, 189, 248, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 34px;
    box-shadow: 0 0 24px rgba(6, 182, 212, 0.35);
    flex-shrink: 0;
}

.appliance-label-sub {
    font-size: 0.75rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 2px;
}

.appliance-name-main {
    font-family: 'Outfit', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.2;
    margin-bottom: 4px;
}

.appliance-type-desc {
    font-size: 0.84rem;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
}

.ai-confidence-panel {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px 24px;
}

.conf-header-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 10px;
}

.conf-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.conf-score-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #34d399;
}

.conf-track-bar {
    width: 100%;
    height: 10px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 10px;
}

.conf-fill-bar {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #06b6d4 0%, #10b981 100%);
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.6);
    transition: width 0.6s ease;
}

.conf-footer-text {
    font-size: 0.76rem;
    color: #64748b;
    display: flex;
    justify-content: space-between;
}

.conf-footer-text span {
    color: #cbd5e1;
}

/* Standby / Low Load Banner */
.standby-box {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(15, 23, 42, 0.85));
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 16px;
    padding: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    margin: 18px 0;
}

.standby-icon-circle {
    width: 58px;
    height: 58px;
    border-radius: 16px;
    background: rgba(245, 158, 11, 0.18);
    border: 1px solid rgba(245, 158, 11, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    flex-shrink: 0;
}

.standby-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: #fbbf24;
    margin-bottom: 4px;
}

.standby-desc {
    font-size: 0.88rem;
    color: #cbd5e1;
}

/* ----------------------------------------------------
   MESCOM BILL INVOICE CARD
   ---------------------------------------------------- */
.mescom-invoice-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(30, 41, 59, 0.8) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.4);
    margin: 20px 0;
}

.invoice-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 16px;
}

.invoice-brand {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: #38bdf8;
    display: flex;
    align-items: center;
    gap: 8px;
}

.invoice-karnataka-pill {
    background: rgba(56, 189, 248, 0.14);
    color: #7dd3fc;
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 0.72rem;
    font-weight: 700;
}

.invoice-net-highlight {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 182, 212, 0.12));
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 14px;
    padding: 16px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 16px;
}

.invoice-net-label {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 0.96rem;
    color: #e2e8f0;
}

.invoice-net-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.85rem;
    font-weight: 800;
    color: #34d399;
    text-shadow: 0 0 16px rgba(16, 185, 129, 0.4);
}

/* Sidebar Brand Banner */
.sidebar-brand-card {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(99, 102, 241, 0.12));
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.sidebar-brand-icon {
    font-size: 24px;
    background: rgba(6, 182, 212, 0.25);
    border-radius: 10px;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(56, 189, 248, 0.4);
}

.sidebar-brand-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 1.05rem;
    color: #f8fafc;
    line-height: 1.1;
}

.sidebar-brand-subtitle {
    font-size: 0.72rem;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""")


# =====================================================
# 4. HELPER FUNCTIONS: UI RENDERING
# =====================================================

def get_appliance_meta(appliance_name):
    """Map appliance name to descriptive icon, type, and theme styling."""
    name_clean = str(appliance_name).lower().strip()
    if "fan" in name_clean:
        return {"icon": "🌀", "color": "#06b6d4", "type": "Inductive Motor Load", "name": "Table Fan"}
    elif "laptop" in name_clean:
        return {"icon": "💻", "color": "#818cf8", "type": "High-Efficiency SMPS Load", "name": "Laptop Charger"}
    elif "phone" in name_clean or "mobile" in name_clean:
        return {"icon": "📱", "color": "#10b981", "type": "Low-Power Adaptive Charger", "name": "Mobile Charger"}
    elif "iron" in name_clean:
        return {"icon": "👔", "color": "#f59e0b", "type": "High-Wattage Resistive Load", "name": "Electric Iron Box"}
    elif "kettle" in name_clean:
        return {"icon": "🫖", "color": "#ec4899", "type": "Thermal Heating Element", "name": "Electric Kettle"}
    elif "refrigerator" in name_clean or "fridge" in name_clean:
        return {"icon": "❄️", "color": "#38bdf8", "type": "Compressor Motor Unit", "name": "Refrigerator"}
    elif "mixer" in name_clean or "grinder" in name_clean:
        return {"icon": "🌪️", "color": "#f97316", "type": "Universal Motor Load", "name": "Mixer Grinder"}
    elif "washing" in name_clean or "washer" in name_clean:
        return {"icon": "🧺", "color": "#0ea5e9", "type": "Inverter Drum Motor", "name": "Washing Machine"}
    elif "bulb" in name_clean or "light" in name_clean:
        return {"icon": "💡", "color": "#fbbf24", "type": "Lighting Luminaire", "name": "LED / Incandescent Lamp"}
    elif "tv" in name_clean or "television" in name_clean:
        return {"icon": "📺", "color": "#a855f7", "type": "Entertainment Display SMPS", "name": "Television"}
    else:
        return {"icon": "⚡", "color": "#38bdf8", "type": "Active Electrical Load", "name": str(appliance_name).title()}


def render_hero_banner():
    """Render high-tech Command Center hero banner."""
    render_html("""
    <div class="hero-container">
        <div class="hero-pill-row">
            <span class="pill-item pill-online"><span class="pulse-radar-dot" style="width:8px; height:8px;"></span> SYSTEM ONLINE</span>
            <span class="pill-item pill-tech">IoT NILM Engine v2.4</span>
            <span class="pill-item pill-grid">Karnataka MESCOM LT-1</span>
        </div>
        <div class="hero-title-wrap">
            <div class="hero-logo-box">⚡</div>
            <div>
                <h1 class="hero-title-text">AI-Enabled Smart Electricity Monitoring</h1>
                <p class="hero-subtitle">Real-time IoT Telemetry • Intelligent Appliance Disaggregation (NILM) • Energy Analytics</p>
            </div>
        </div>
        <div class="hero-arch-row">
            <div class="arch-chip">MCU: <strong>ESP32</strong></div>
            <div class="arch-chip">Sensor: <strong>PZEM-004T AC</strong></div>
            <div class="arch-chip">Comms: <strong>LoRa + Wi-Fi</strong></div>
            <div class="arch-chip">Cloud: <strong>Firebase Firestore</strong></div>
            <div class="arch-chip">AI Engine: <strong>Multi-Model Ensemble</strong></div>
        </div>
    </div>
    """)


def render_telemetry_status(mode_str, project_id, doc_id="Active", timestamp_str=None):
    """Render live connection status bar."""
    if not timestamp_str:
        timestamp_str = "Awaiting Packet"

    render_html(f"""
    <div class="telemetry-status-bar">
        <div class="status-left">
            <div class="pulse-radar-dot"></div>
            <div>
                <div class="status-title-text">{mode_str} — TELEMETRY STREAM ACTIVE</div>
                <div class="status-desc-text">ESP32 & PZEM-004T streaming electrical telemetry via Flask and Firebase.</div>
            </div>
        </div>
        <div class="status-chips-wrap">
            <div class="status-chip">Project: <strong>{project_id}</strong></div>
            <div class="status-chip">Doc ID: <strong>{doc_id[:12]}...</strong></div>
            <div class="status-chip">Sync: <strong>{timestamp_str}</strong></div>
        </div>
    </div>
    """)


def render_sensor_telemetry_grid(v, i, p, energy_str, freq, pf):
    """Render modern glassmorphic sensor telemetry cards."""
    # Voltage stability indicator
    v_tag = "STABLE" if 215.0 <= v <= 245.0 else ("UNDERVOLTAGE" if v < 215.0 else "OVERVOLTAGE")
    v_tag_cls = "tag-green" if v_tag == "STABLE" else "tag-amber"

    # Current load indicator
    i_tag = "NO LOAD" if i < 0.05 else ("NORMAL" if i <= 5.0 else "HIGH LOAD")
    i_tag_cls = "tag-cyan" if i_tag != "HIGH LOAD" else "tag-amber"

    # Power status
    p_tag = "ACTIVE DRAW" if p >= 1.0 else "QUIESCENT"
    p_tag_cls = "tag-amber" if p >= 1.0 else "tag-green"

    # PF Efficiency
    pf_tag = "OPTIMAL" if pf >= 0.90 else ("ACCEPTABLE" if pf >= 0.80 else "LOW PF")
    pf_tag_cls = "tag-green" if pf >= 0.90 else "tag-amber"

    render_html(f"""
    <div class="telemetry-grid">
        <!-- Voltage -->
        <div class="sensor-card">
            <div class="card-top">
                <div class="card-icon-box icon-cyan-box">⚡</div>
                <span class="card-tag {v_tag_cls}">{v_tag}</span>
            </div>
            <div>
                <div class="card-label-text">VOLTAGE</div>
                <div class="card-value-display">{v:.2f}<span class="card-unit-text">V</span></div>
            </div>
            <div class="card-footer-info">
                <span>Grid Nominal</span>
                <strong>230.00 V AC</strong>
            </div>
        </div>

        <!-- Current -->
        <div class="sensor-card">
            <div class="card-top">
                <div class="card-icon-box icon-emerald-box">🔌</div>
                <span class="card-tag {i_tag_cls}">{i_tag}</span>
            </div>
            <div>
                <div class="card-label-text">CURRENT</div>
                <div class="card-value-display">{i:.3f}<span class="card-unit-text">A</span></div>
            </div>
            <div class="card-footer-info">
                <span>Line Current</span>
                <strong>PZEM Shunt</strong>
            </div>
        </div>

        <!-- Power -->
        <div class="sensor-card">
            <div class="card-top">
                <div class="card-icon-box icon-amber-box">🔥</div>
                <span class="card-tag {p_tag_cls}">{p_tag}</span>
            </div>
            <div>
                <div class="card-label-text">ACTIVE POWER</div>
                <div class="card-value-display">{p:.2f}<span class="card-unit-text">W</span></div>
            </div>
            <div class="card-footer-info">
                <span>Real Work</span>
                <strong>V × I × cos φ</strong>
            </div>
        </div>

        <!-- Energy -->
        <div class="sensor-card">
            <div class="card-top">
                <div class="card-icon-box icon-violet-box">🔋</div>
                <span class="card-tag tag-violet">METER</span>
            </div>
            <div>
                <div class="card-label-text">TOTAL ENERGY</div>
                <div class="card-value-display">{energy_str}<span class="card-unit-text">kWh</span></div>
            </div>
            <div class="card-footer-info">
                <span>Accumulator</span>
                <strong>Cumulative</strong>
            </div>
        </div>
    </div>

    <div class="telemetry-grid-secondary">
        <!-- Frequency -->
        <div class="sensor-card">
            <div class="card-top">
                <div class="card-icon-box icon-teal-box">〰️</div>
                <span class="card-tag tag-green">GRID SYNC</span>
            </div>
            <div>
                <div class="card-label-text">GRID FREQUENCY</div>
                <div class="card-value-display">{freq:.2f}<span class="card-unit-text">Hz</span></div>
            </div>
            <div class="card-footer-info">
                <span>Nominal Grid</span>
                <strong>50.00 Hz ± 0.5%</strong>
            </div>
        </div>

        <!-- Power Factor -->
        <div class="sensor-card">
            <div class="card-top">
                <div class="card-icon-box icon-blue-box">🎯</div>
                <span class="card-tag {pf_tag_cls}">{pf_tag}</span>
            </div>
            <div>
                <div class="card-label-text">POWER FACTOR</div>
                <div class="card-value-display">{pf:.2f}<span class="card-unit-text">cos φ</span></div>
            </div>
            <div class="card-footer-info">
                <span>Efficiency Target</span>
                <strong>> 0.90 Target</strong>
            </div>
        </div>
    </div>
    """)


def render_ai_prediction_card(prediction, score, model_choice, power_w):
    """Render high-tech AI NILM appliance identification card."""
    if power_w < 1.0:
        render_html(f"""
        <div class="standby-box">
            <div class="standby-icon-circle">💤</div>
            <div>
                <div class="standby-title">STANDBY / NO LOAD DETECTED</div>
                <div class="standby-desc">
                    Active power draw is <strong>{power_w:.2f} W</strong> (below 1.0 W threshold). 
                    No active appliance load signature is currently consuming meaningful current.
                </div>
            </div>
        </div>
        """)
        return

    meta = get_appliance_meta(prediction)

    render_html(f"""
    <div class="ai-showcase-panel">
        <div class="ai-header-flex">
            <div class="ai-title-wrap">
                <span class="ai-chip-pill">⚡ Neural NILM Engine</span>
                <span style="font-size:0.88rem; color:#cbd5e1; font-weight:600;">Appliance Identification</span>
            </div>
            <div class="ai-model-tag-pill">
                Active Classifier: <strong>{model_choice}</strong>
            </div>
        </div>

        <div class="ai-content-grid">
            <div class="appliance-identity-card">
                <div class="appliance-icon-ring">
                    {meta['icon']}
                </div>
                <div>
                    <div class="appliance-label-sub">DETECTED APPLIANCE</div>
                    <div class="appliance-name-main">{meta['name']}</div>
                    <div class="appliance-type-desc">Load Profile: {meta['type']}</div>
                </div>
            </div>

            <div class="ai-confidence-panel">
                <div class="conf-header-row">
                    <span class="conf-label">Model Confidence</span>
                    <span class="conf-score-num">{score:.2f}%</span>
                </div>
                <div class="conf-track-bar">
                    <div class="conf-fill-bar" style="width: {min(100.0, max(5.0, score))}%;"></div>
                </div>
                <div class="conf-footer-text">
                    <span>Signature: {power_w:.1f} W</span>
                    <span>Status: High Probability Match</span>
                </div>
            </div>
        </div>
    </div>
    """)


# =====================================================
# 5. LOAD MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Selected model file not found: {model_path.name}. "
            "Run python scripts/train_compare_models.py first."
        )
    return joblib.load(model_path)


# =====================================================
# 6. LOAD OFFLINE DATASET
# =====================================================

@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError("Dataset not found.")
    return pd.read_csv(DATASET_PATH)


# =====================================================
# 7. FIREBASE CONNECTION
# =====================================================

@st.cache_resource
def connect_firebase():
    if not FIREBASE_KEY_PATH.exists():
        raise FileNotFoundError("Firebase service account file not found.")

    if EXPECTED_PROJECT_ID == "SET_YOUR_FIREBASE_PROJECT_ID":
        raise ValueError("Set FIREBASE_PROJECT_ID in your .env file.")

    cred = credentials.Certificate(str(FIREBASE_KEY_PATH))

    if cred.project_id != EXPECTED_PROJECT_ID:
        raise ValueError(
            f"Firebase project mismatch. Credentials belong to '{cred.project_id}', "
            f"but FIREBASE_PROJECT_ID is '{EXPECTED_PROJECT_ID}'."
        )

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)

    return firestore.client()


# =====================================================
# 8. GET LATEST FIREBASE READING
# =====================================================

def get_latest_reading():
    db = connect_firebase()
    docs = (
        db.collection(COLLECTION_NAME)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )
    for doc in docs:
        reading = doc.to_dict()
        reading["document_id"] = doc.id
        return reading
    return None


# =====================================================
# 9. GET HISTORICAL FIREBASE READINGS
# =====================================================

@st.cache_data(ttl=60)
def get_historical_readings(limit=500):
    db = connect_firebase()
    docs = (
        db.collection(COLLECTION_NAME)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )

    records = []
    for doc in docs:
        record = doc.to_dict()
        record["document_id"] = doc.id
        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    numeric_columns = [
        "voltage_V",
        "current_A",
        "power_W",
        "energy_kWh",
        "frequency_Hz",
        "power_factor"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert Firebase UTC timestamps to IST
    df["timestamp"] = df["timestamp"].dt.tz_convert(TIMEZONE)
    df = df.sort_values("timestamp")
    return df


# =====================================================
# 10. MACHINE LEARNING PREDICTION
# =====================================================

def predict_appliance(model, reading):
    input_data = pd.DataFrame(
        [[float(reading[f]) for f in FEATURES]],
        columns=FEATURES
    )
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    score = max(probabilities) * 100
    return prediction, score


# =====================================================
# 11. HELPER: PREPARE IST GRAPH DATA
# =====================================================

def prepare_ist_graph_data(df):
    graph_df = df.copy()
    graph_df["timestamp"] = pd.to_datetime(graph_df["timestamp"], utc=True, errors="coerce")
    graph_df["timestamp"] = (
        graph_df["timestamp"]
        .dt.tz_convert(TIMEZONE)
        .dt.tz_localize(None)
    )
    graph_df = graph_df.dropna(subset=["timestamp"])
    return graph_df.sort_values("timestamp")


# =====================================================
# 12. ADVANCED ELECTRICITY ANALYTICS & MONITORING
# =====================================================

def display_electricity_analytics(df, model=None):
    if df.empty:
        st.warning("No historical readings are available yet.")
        return

    df = prepare_ist_graph_data(df)

    # =================================================
    # 1. ELECTRICAL MONITORING
    # =================================================
    st.divider()
    render_html("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <span style="font-size:1.3rem;">⚡</span>
        <div>
            <h3 style="margin:0; font-family:'Outfit',sans-serif; color:#f8fafc;">Electrical Monitoring</h3>
            <div style="font-size:0.8rem; color:#94a3b8;">Real-time voltage stability, line current and AC grid frequency telemetry.</div>
        </div>
    </div>
    """)

    # --- Active Power Curve ---
    if "power_W" in df.columns:
        power_df = df[["timestamp", "power_W"]].copy()
        power_df["power_W"] = pd.to_numeric(power_df["power_W"], errors="coerce").dropna()

        if not power_df.empty:
            st.markdown("##### ⚡ Power Consumption Curve")
            power_chart = (
                alt.Chart(power_df)
                .mark_area(
                    line={"color": "#06b6d4", "width": 2.5},
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[
                            alt.GradientStop(color="rgba(6, 182, 212, 0.45)", offset=0),
                            alt.GradientStop(color="rgba(6, 182, 212, 0.02)", offset=1)
                        ],
                        x1=1, x2=1, y1=1, y2=0
                    )
                )
                .encode(
                    x=alt.X("timestamp:T", title="Time (IST)", axis=alt.Axis(format="%H:%M", labelAngle=0, gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8", titleColor="#cbd5e1")),
                    y=alt.Y("power_W:Q", title="Active Power (W)", scale=alt.Scale(zero=True), axis=alt.Axis(gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8", titleColor="#cbd5e1")),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Time (IST)", format="%d-%b %H:%M:%S"),
                        alt.Tooltip("power_W:Q", title="Active Power (W)", format=".2f")
                    ]
                )
                .properties(height=260, background="transparent")
                .configure_view(strokeWidth=0)
                .interactive()
            )
            st.altair_chart(power_chart, use_container_width=True)
            st.caption(f"Peak Power: **{power_df['power_W'].max():.2f} W** | Average Power: **{power_df['power_W'].mean():.2f} W**")

    # --- Voltage & Current Row ---
    col1, col2 = st.columns(2)

    if "voltage_V" in df.columns:
        voltage_df = df[["timestamp", "voltage_V"]].copy()
        voltage_df["voltage_V"] = pd.to_numeric(voltage_df["voltage_V"], errors="coerce").dropna()

        with col1:
            st.markdown("##### 🔌 Voltage Stability")
            if not voltage_df.empty:
                voltage_chart = (
                    alt.Chart(voltage_df)
                    .mark_area(
                        line={"color": "#818cf8", "width": 2},
                        color=alt.Gradient(
                            gradient="linear",
                            stops=[
                                alt.GradientStop(color="rgba(129, 140, 248, 0.35)", offset=0),
                                alt.GradientStop(color="rgba(129, 140, 248, 0.02)", offset=1)
                            ],
                            x1=1, x2=1, y1=1, y2=0
                        )
                    )
                    .encode(
                        x=alt.X("timestamp:T", title="Time (IST)", axis=alt.Axis(format="%H:%M", labelAngle=0, gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8", titleColor="#cbd5e1")),
                        y=alt.Y("voltage_V:Q", title="Voltage (V)", scale=alt.Scale(zero=False), axis=alt.Axis(gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8", titleColor="#cbd5e1")),
                        tooltip=[
                            alt.Tooltip("timestamp:T", title="Time (IST)", format="%d-%b %H:%M:%S"),
                            alt.Tooltip("voltage_V:Q", title="Voltage (V)", format=".2f")
                        ]
                    )
                    .properties(height=240, background="transparent")
                    .configure_view(strokeWidth=0)
                    .interactive()
                )
                st.altair_chart(voltage_chart, use_container_width=True)
                st.caption(f"Min: {voltage_df['voltage_V'].min():.2f} V | Max: {voltage_df['voltage_V'].max():.2f} V | Avg: {voltage_df['voltage_V'].mean():.2f} V")

    if "current_A" in df.columns:
        current_df = df[["timestamp", "current_A"]].copy()
        current_df["current_A"] = pd.to_numeric(current_df["current_A"], errors="coerce").dropna()

        with col2:
            st.markdown("##### 💡 Load Current")
            if not current_df.empty:
                current_chart = (
                    alt.Chart(current_df)
                    .mark_area(
                        line={"color": "#34d399", "width": 2},
                        color=alt.Gradient(
                            gradient="linear",
                            stops=[
                                alt.GradientStop(color="rgba(52, 211, 153, 0.35)", offset=0),
                                alt.GradientStop(color="rgba(52, 211, 153, 0.02)", offset=1)
                            ],
                            x1=1, x2=1, y1=1, y2=0
                        )
                    )
                    .encode(
                        x=alt.X("timestamp:T", title="Time (IST)", axis=alt.Axis(format="%H:%M", labelAngle=0, gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8", titleColor="#cbd5e1")),
                        y=alt.Y("current_A:Q", title="Current (A)", scale=alt.Scale(zero=True), axis=alt.Axis(gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8", titleColor="#cbd5e1")),
                        tooltip=[
                            alt.Tooltip("timestamp:T", title="Time (IST)", format="%d-%b %H:%M:%S"),
                            alt.Tooltip("current_A:Q", title="Current (A)", format=".3f")
                        ]
                    )
                    .properties(height=240, background="transparent")
                    .configure_view(strokeWidth=0)
                    .interactive()
                )
                st.altair_chart(current_chart, use_container_width=True)
                st.caption(f"Max: {current_df['current_A'].max():.3f} A | Avg: {current_df['current_A'].mean():.3f} A")

    # --- NEW: Frequency Monitoring ---
    if "frequency_Hz" in df.columns:
        freq_df = df[["timestamp", "frequency_Hz"]].copy()
        freq_df["frequency_Hz"] = pd.to_numeric(freq_df["frequency_Hz"], errors="coerce").dropna()

        if not freq_df.empty:
            st.markdown("##### 〰️ Frequency Monitoring (New)")
            min_f = freq_df["frequency_Hz"].min()
            max_f = freq_df["frequency_Hz"].max()
            f_domain = [max(45.0, min_f - 0.5), min(55.0, max_f + 0.5)]

            freq_chart = (
                alt.Chart(freq_df)
                .mark_area(
                    line={"color": "#14b8a6", "width": 2},
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[
                            alt.GradientStop(color="rgba(20, 184, 166, 0.35)", offset=0),
                            alt.GradientStop(color="rgba(20, 184, 166, 0.02)", offset=1)
                        ],
                        x1=1, x2=1, y1=1, y2=0
                    )
                )
                .encode(
                    x=alt.X("timestamp:T", title="Time (IST)", axis=alt.Axis(format="%H:%M", labelAngle=0, gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8", titleColor="#cbd5e1")),
                    y=alt.Y("frequency_Hz:Q", title="Grid Frequency (Hz)", scale=alt.Scale(zero=False, domain=f_domain), axis=alt.Axis(gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8", titleColor="#cbd5e1")),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Time (IST)", format="%d-%b %H:%M:%S"),
                        alt.Tooltip("frequency_Hz:Q", title="Frequency (Hz)", format=".2f")
                    ]
                )
                .properties(height=240, background="transparent")
                .configure_view(strokeWidth=0)
                .interactive()
            )
            st.altair_chart(freq_chart, use_container_width=True)
            st.caption(f"Nominal: **50.00 Hz** | Min: **{min_f:.2f} Hz** | Max: **{max_f:.2f} Hz** | Status: **Grid Frequency Synchronized**")


    # =================================================
    # 2. POWER QUALITY ANALYSIS
    # =================================================
    st.divider()
    render_html("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <span style="font-size:1.3rem;">🎯</span>
        <div>
            <h3 style="margin:0; font-family:'Outfit',sans-serif; color:#f8fafc;">Power Quality Analysis</h3>
            <div style="font-size:0.8rem; color:#94a3b8;">Active vs apparent power disaggregation and power factor efficiency trends.</div>
        </div>
    </div>
    """)

    pq_col1, pq_col2 = st.columns(2)

    # --- NEW: Active vs Apparent Power ---
    with pq_col1:
        st.markdown("##### ⚡ Active vs Apparent Power (New)")
        if "voltage_V" in df.columns and "current_A" in df.columns and "power_W" in df.columns:
            pq_df = df[["timestamp", "voltage_V", "current_A", "power_W"]].copy()
            pq_df["power_W"] = pd.to_numeric(pq_df["power_W"], errors="coerce")
            pq_df["apparent_power_VA"] = pd.to_numeric(pq_df["voltage_V"], errors="coerce") * pd.to_numeric(pq_df["current_A"], errors="coerce")
            pq_df = pq_df.dropna(subset=["timestamp", "power_W", "apparent_power_VA"])

            if not pq_df.empty:
                melted_p = pq_df.melt(
                    id_vars=["timestamp"],
                    value_vars=["power_W", "apparent_power_VA"],
                    var_name="Power_Metric",
                    value_name="Value"
                )
                melted_p["Power_Metric"] = melted_p["Power_Metric"].replace({
                    "power_W": "Active Power P (W)",
                    "apparent_power_VA": "Apparent Power S (VA)"
                })

                p_vs_s_chart = (
                    alt.Chart(melted_p)
                    .mark_line(strokeWidth=2.2)
                    .encode(
                        x=alt.X("timestamp:T", title="Time (IST)", axis=alt.Axis(format="%H:%M", labelAngle=0, gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8", titleColor="#cbd5e1")),
                        y=alt.Y("Value:Q", title="Power (W / VA)", scale=alt.Scale(zero=True), axis=alt.Axis(gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8", titleColor="#cbd5e1")),
                        color=alt.Color("Power_Metric:N", title="Power Component", scale=alt.Scale(domain=["Active Power P (W)", "Apparent Power S (VA)"], range=["#06b6d4", "#f59e0b"]), legend=alt.Legend(orient="top", labelColor="#cbd5e1", titleColor="#94a3b8")),
                        tooltip=[
                            alt.Tooltip("timestamp:T", title="Time (IST)", format="%d-%b %H:%M:%S"),
                            alt.Tooltip("Power_Metric:N", title="Component"),
                            alt.Tooltip("Value:Q", title="Magnitude", format=".2f")
                        ]
                    )
                    .properties(height=260, background="transparent")
                    .configure_view(strokeWidth=0)
                    .interactive()
                )
                st.altair_chart(p_vs_s_chart, use_container_width=True)
                peak_p = pq_df["power_W"].max()
                peak_s = pq_df["apparent_power_VA"].max()
                st.caption(f"Peak Active (P): **{peak_p:.2f} W** | Peak Apparent (S): **{peak_s:.2f} VA**")

    # --- NEW: Power Factor Trend ---
    with pq_col2:
        st.markdown("##### 🎯 Power Factor Trend (New)")
        if "power_factor" in df.columns:
            pf_df = df[["timestamp", "power_factor"]].copy()
            pf_df["power_factor"] = pd.to_numeric(pf_df["power_factor"], errors="coerce").dropna()

            if not pf_df.empty:
                pf_chart = (
                    alt.Chart(pf_df)
                    .mark_area(
                        line={"color": "#818cf8", "width": 2},
                        color=alt.Gradient(
                            gradient="linear",
                            stops=[
                                alt.GradientStop(color="rgba(129, 140, 248, 0.35)", offset=0),
                                alt.GradientStop(color="rgba(129, 140, 248, 0.02)", offset=1)
                            ],
                            x1=1, x2=1, y1=1, y2=0
                        )
                    )
                    .encode(
                        x=alt.X("timestamp:T", title="Time (IST)", axis=alt.Axis(format="%H:%M", labelAngle=0, gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8", titleColor="#cbd5e1")),
                        y=alt.Y("power_factor:Q", title="Power Factor (cos φ)", scale=alt.Scale(domain=[0.0, 1.0]), axis=alt.Axis(gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8", titleColor="#cbd5e1")),
                        tooltip=[
                            alt.Tooltip("timestamp:T", title="Time (IST)", format="%d-%b %H:%M:%S"),
                            alt.Tooltip("power_factor:Q", title="Power Factor", format=".2f")
                        ]
                    )
                    .properties(height=260, background="transparent")
                    .configure_view(strokeWidth=0)
                    .interactive()
                )
                st.altair_chart(pf_chart, use_container_width=True)
                mean_pf = pf_df["power_factor"].mean()
                pf_status = "High Efficiency" if mean_pf >= 0.90 else ("Acceptable" if mean_pf >= 0.80 else "Poor Power Factor")
                st.caption(f"Average PF: **{mean_pf:.2f}** | Benchmark: **> 0.90** | Quality: **{pf_status}**")


    # =================================================
    # 3. ENERGY ANALYTICS
    # =================================================
    st.divider()
    render_html("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <span style="font-size:1.3rem;">📊</span>
        <div>
            <h3 style="margin:0; font-family:'Outfit',sans-serif; color:#f8fafc;">Energy Analytics</h3>
            <div style="font-size:0.8rem; color:#94a3b8;">Daily energy consumption totals and appliance-wise distribution breakdowns.</div>
        </div>
    </div>
    """)

    ea_col1, ea_col2 = st.columns(2)

    # --- NEW: Daily Energy Consumption ---
    with ea_col1:
        st.markdown("##### 📈 Daily Energy Consumption (New)")
        daily_energy = pd.DataFrame()

        if "energy_kWh" in df.columns and not df["energy_kWh"].dropna().empty:
            e_df = df[["timestamp", "energy_kWh"]].dropna().sort_values("timestamp").copy()
            e_df["energy_used"] = e_df["energy_kWh"].diff()
            e_df.loc[e_df["energy_used"] < 0, "energy_used"] = 0
            e_df = e_df.dropna(subset=["energy_used"])
            if not e_df.empty and e_df["energy_used"].sum() > 0:
                e_df["date"] = e_df["timestamp"].dt.strftime("%d-%b-%Y")
                daily_energy = e_df.groupby("date")["energy_used"].sum().reset_index()

        # Fallback to trapezoidal integration if energy_kWh is not cumulative or unavailable
        if daily_energy.empty and "power_W" in df.columns:
            p_df = df[["timestamp", "power_W"]].dropna().sort_values("timestamp").copy()
            if not p_df.empty:
                dt_h = p_df["timestamp"].diff().dt.total_seconds().fillna(1.0).clip(0, 300) / 3600.0
                avg_p = (p_df["power_W"] + p_df["power_W"].shift(1).fillna(p_df["power_W"])) / 2.0
                p_df["energy_used"] = (avg_p * dt_h) / 1000.0
                p_df["date"] = p_df["timestamp"].dt.strftime("%d-%b-%Y")
                daily_energy = p_df.groupby("date")["energy_used"].sum().reset_index()

        if not daily_energy.empty:
            daily_chart = (
                alt.Chart(daily_energy)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#06b6d4")
                .encode(
                    x=alt.X("date:N", title="Date (IST)", sort=None, axis=alt.Axis(labelColor="#94a3b8", labelAngle=-20, titleColor="#cbd5e1")),
                    y=alt.Y("energy_used:Q", title="Energy (kWh)", scale=alt.Scale(zero=True), axis=alt.Axis(gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8", titleColor="#cbd5e1")),
                    tooltip=[
                        alt.Tooltip("date:N", title="Date (IST)"),
                        alt.Tooltip("energy_used:Q", title="Energy (kWh)", format=".4f")
                    ]
                )
                .properties(height=260, background="transparent")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(daily_chart, use_container_width=True)
            tot_e = daily_energy["energy_used"].sum()
            st.caption(f"Total Cumulative Energy: **{tot_e:.4f} kWh**")
        else:
            st.info("Insufficient readings to aggregate daily energy.")

    # --- NEW: Appliance-wise Energy Distribution ---
    with ea_col2:
        st.markdown("##### 🥧 Appliance-wise Energy Distribution (New)")
        app_df = df.copy()

        # If appliance column is missing or empty, predict using active ML model!
        if "appliance" not in app_df.columns or app_df["appliance"].isna().all():
            if model is not None:
                try:
                    feat_data = app_df[FEATURES].fillna(0)
                    app_df["appliance"] = model.predict(feat_data)
                    app_df.loc[app_df["power_W"] < 1.0, "appliance"] = "Standby / Quiescent"
                except Exception:
                    app_df["appliance"] = "General AC Load"
            else:
                app_df["appliance"] = "Active Load"

        app_summary = (
            app_df.groupby("appliance")
            .agg(
                total_power=("power_W", "sum"),
                avg_power=("power_W", "mean"),
                reading_count=("power_W", "count")
            )
            .reset_index()
        )
        total_p_all = max(app_summary["total_power"].sum(), 0.001)
        app_summary["share_pct"] = (app_summary["total_power"] / total_p_all) * 100.0
        app_summary = app_summary.sort_values("total_power", ascending=False)

        if not app_summary.empty:
            donut_chart = (
                alt.Chart(app_summary)
                .mark_arc(innerRadius=55, stroke="#070b14", strokeWidth=2)
                .encode(
                    theta=alt.Theta("total_power:Q", title="Energy Share"),
                    color=alt.Color("appliance:N", title="Appliance", scale=alt.Scale(scheme="category10"), legend=alt.Legend(orient="right", labelColor="#cbd5e1", titleColor="#94a3b8")),
                    tooltip=[
                        alt.Tooltip("appliance:N", title="Appliance"),
                        alt.Tooltip("share_pct:Q", title="Share (%)", format=".1f"),
                        alt.Tooltip("avg_power:Q", title="Avg Power (W)", format=".1f"),
                        alt.Tooltip("reading_count:Q", title="Samples")
                    ]
                )
                .properties(height=260, background="transparent")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(donut_chart, use_container_width=True)
            top_app = app_summary.iloc[0]["appliance"]
            top_share = app_summary.iloc[0]["share_pct"]
            st.caption(f"Primary Consumer: **{top_app}** (**{top_share:.1f}%** of total load)")
        else:
            st.info("Appliance distribution data unavailable.")


# =====================================================
# 13. MESCOM BILL ESTIMATOR
# =====================================================

def display_mescom_bill_estimator():
    st.divider()
    st.markdown("### 🏠 MESCOM Household Electricity Bill Estimator")
    st.caption("Karnataka Electricity Regulatory Commission (KERC) • LT-1 Domestic Tariff Model")

    c1, c2 = st.columns(2)
    sanctioned_load = c1.number_input("Sanctioned Load (kW)", min_value=0.5, max_value=50.0, value=2.0, step=0.5)
    monthly_units = c2.number_input("Monthly Consumption (Units / kWh)", min_value=0.0, value=57.0, step=1.0)

    with st.expander("⚙️ View / Adjust MESCOM Tariff Rates", expanded=False):
        col1, col2 = st.columns(2)
        fixed_rate = col1.number_input("Fixed Charge (₹ / kW)", min_value=0.0, value=145.0, step=1.0)
        energy_rate = col2.number_input("Energy Charge (₹ / Unit)", min_value=0.0, value=5.80, step=0.01)
        fppca_rate = col1.number_input("FPPCA (₹ / Unit)", min_value=0.0, value=0.37, step=0.01)
        pg_rate = col2.number_input("P&G Charge (₹ / Unit)", min_value=0.0, value=0.36, step=0.01)
        tax_rate = st.number_input("Electricity Tax (%)", min_value=0.0, max_value=100.0, value=9.0, step=0.5)

    fixed_charge = sanctioned_load * fixed_rate
    energy_charge = monthly_units * energy_rate
    fppca_charge = monthly_units * fppca_rate
    pg_charge = monthly_units * pg_rate
    electricity_tax = energy_charge * tax_rate / 100.0
    gross_bill = fixed_charge + energy_charge + fppca_charge + pg_charge + electricity_tax

    eligible_for_subsidy = st.checkbox("Apply Gruha Jyothi / Government Subsidy Credit", value=False)
    if eligible_for_subsidy:
        subsidy_amount = st.number_input("Subsidy / Credit Amount (₹)", min_value=0.0, max_value=float(gross_bill), value=0.0, step=10.0)
    else:
        subsidy_amount = 0.0

    estimated_bill_after_subsidy = max(0.0, gross_bill - subsidy_amount)

    # Render Modern Invoice Card with clean render_html
    render_html(f"""
    <div class="mescom-invoice-card">
        <div class="invoice-header">
            <div class="invoice-brand">
                ⚡ MESCOM TARIFF ESTIMATE
            </div>
            <span class="invoice-karnataka-pill">KARNATAKA LT-1</span>
        </div>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-bottom:14px;">
            <div style="background:rgba(15,23,42,0.6); padding:12px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.06);">
                <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Fixed Charges</div>
                <div style="font-family:'JetBrains Mono', monospace; font-size:1.1rem; color:#f1f5f9; font-weight:700;">₹ {fixed_charge:.2f}</div>
            </div>
            <div style="background:rgba(15,23,42,0.6); padding:12px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.06);">
                <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Energy Charges</div>
                <div style="font-family:'JetBrains Mono', monospace; font-size:1.1rem; color:#f1f5f9; font-weight:700;">₹ {energy_charge:.2f}</div>
            </div>
            <div style="background:rgba(15,23,42,0.6); padding:12px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.06);">
                <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">FPPCA + P&G</div>
                <div style="font-family:'JetBrains Mono', monospace; font-size:1.1rem; color:#f1f5f9; font-weight:700;">₹ {(fppca_charge + pg_charge):.2f}</div>
            </div>
            <div style="background:rgba(15,23,42,0.6); padding:12px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.06);">
                <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Electricity Tax (9%)</div>
                <div style="font-family:'JetBrains Mono', monospace; font-size:1.1rem; color:#f1f5f9; font-weight:700;">₹ {electricity_tax:.2f}</div>
            </div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 4px; font-size:0.86rem; color:#94a3b8;">
            <span>Gross Calculated Amount:</span>
            <span style="font-family:'JetBrains Mono', monospace; color:#cbd5e1;">₹ {gross_bill:.2f}</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 4px; font-size:0.86rem; color:#34d399;">
            <span>Applied Subsidy Relief:</span>
            <span style="font-family:'JetBrains Mono', monospace;">- ₹ {subsidy_amount:.2f}</span>
        </div>
        <div class="invoice-net-highlight">
            <div>
                <div class="invoice-net-label">NET ESTIMATED MONTHLY BILL</div>
                <div style="font-size:0.75rem; color:#94a3b8;">Includes energy charges, fixed charges, taxes & relief</div>
            </div>
            <div class="invoice-net-val">₹ {estimated_bill_after_subsidy:.2f}</div>
        </div>
    </div>
    """)


# =====================================================
# 14. DASHBOARD HERO HEADER
# =====================================================

render_hero_banner()


# =====================================================
# 15. SIDEBAR CONFIGURATION
# =====================================================

render_html("""
<div class="sidebar-brand-card">
    <div class="sidebar-brand-icon">⚡</div>
    <div>
        <div class="sidebar-brand-title">ENERGY OS</div>
        <div class="sidebar-brand-subtitle">AI & IoT Monitoring</div>
    </div>
</div>
""")

try:
    registry_path = MODEL_DIR / "model_registry.json"
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as registry_file:
            model_registry = json.load(registry_file)
    else:
        model_registry = {"Random Forest (legacy)": "random_forest_appliance.pkl"}

    st.sidebar.markdown("##### 🤖 ML Prediction Model")
    model_choice = st.sidebar.selectbox(
        "Select ML Model",
        list(model_registry.keys()),
        label_visibility="collapsed"
    )
    selected_model_path = MODEL_DIR / model_registry[model_choice]
    model = load_model(selected_model_path)
    st.sidebar.caption(f"✓ Model `{model_choice}` active.")

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("##### 📡 Operating Mode")
mode = st.sidebar.selectbox(
    "Select Operating Mode",
    ["LIVE MODE", "OFFLINE DEMO MODE"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("##### ☁️ Cloud Backend")
render_html(f"""
<div style="background:rgba(15,23,42,0.8); border:1px solid rgba(56,189,248,0.2); border-radius:10px; padding:10px 14px; font-family:'JetBrains Mono', monospace; font-size:0.78rem; color:#38bdf8;">
    🔥 {EXPECTED_PROJECT_ID}
</div>
""")

st.sidebar.markdown("---")
render_html("""
<div style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:10px 12px; font-size:0.75rem; color:#94a3b8;">
    📍 MESCOM Tariff | Karnataka | INR ₹<br>
    ⚡ Single Phase 230V AC System
</div>
""")


# =====================================================
# 16. OFFLINE DEMO MODE
# =====================================================

if mode == "OFFLINE DEMO MODE":
    render_telemetry_status("OFFLINE DEMO MODE", "LOCAL DATASET", "Offline Replay", "Recorded Dataset")

    try:
        df = load_dataset()
    except Exception as e:
        st.error(str(e))
        st.stop()

    required_columns = FEATURES + ["appliance", "session_id"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Dataset is missing columns: {missing_columns}")
        st.stop()

    if df.empty:
        st.warning("The offline dataset is empty.")
        st.stop()

    # Overview Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Readings", f"{len(df):,}")
    c2.metric("Appliance Classes", df["appliance"].nunique())
    c3.metric("Recorded Sessions", df["session_id"].nunique())

    st.markdown("#### 🔍 Select Appliance Dataset Record")
    appliances = sorted(df["appliance"].dropna().unique())
    selected_appliance = st.selectbox("Choose an Appliance to Inspect", appliances)

    filtered_df = df[df["appliance"] == selected_appliance].reset_index(drop=True)
    if filtered_df.empty:
        st.warning("No readings found for this appliance.")
        st.stop()

    selected_row = st.slider("Select Sample Index", 0, len(filtered_df) - 1, 0)
    reading = filtered_df.iloc[selected_row]

    input_reading = {
        feature: float(reading[feature])
        for feature in FEATURES
    }

    # Render Sensor Parameters
    st.markdown("#### ⚡ Recorded Sensor Telemetry")
    render_sensor_telemetry_grid(
        input_reading["voltage_V"],
        input_reading["current_A"],
        input_reading["power_W"],
        f"{float(reading.get('energy_kWh', 0.0)):.4f}" if "energy_kWh" in reading else "N/A",
        input_reading["frequency_Hz"],
        input_reading["power_factor"]
    )

    # AI Prediction
    prediction, score = predict_appliance(model, input_reading)
    render_ai_prediction_card(prediction, score, model_choice, input_reading["power_W"])

    if prediction == selected_appliance:
        st.success(f"✓ AI prediction perfectly matches ground-truth label: `{selected_appliance}`")
    else:
        st.info(f"AI predicted `{prediction}`, whereas dataset ground truth is `{selected_appliance}`")

    # Power Curve
    st.markdown("#### 📈 Appliance Power Signature (First 100 Samples)")
    st.line_chart(filtered_df[["power_W"]].head(100), height=250, use_container_width=True)

    with st.expander("📋 View Raw Dataframe", expanded=False):
        st.dataframe(filtered_df.head(20), use_container_width=True)

    # Historical and Appliance Analytics
    display_electricity_analytics(df, model=model)


# =====================================================
# 17. LIVE MODE
# =====================================================

elif mode == "LIVE MODE":
    col_btn, col_empty = st.columns([2, 5])
    with col_btn:
        if st.button("🔄 Refresh Live Reading", use_container_width=True):
            get_historical_readings.clear()
            st.rerun()

    try:
        reading = get_latest_reading()

        if reading is None:
            st.warning("No sensor readings found in Firebase Firestore collection.")
            st.stop()

        missing = [feature for feature in FEATURES if feature not in reading or reading[feature] is None]
        if missing:
            st.error(f"Missing sensor fields from device: {missing}")
            st.stop()

        input_reading = {
            feature: float(reading[feature])
            for feature in FEATURES
        }

        # Timestamp formatting
        timestamp_value = reading.get("timestamp")
        timestamp_display = "Live"
        if timestamp_value is not None:
            timestamp_ist = pd.to_datetime(timestamp_value, utc=True, errors="coerce")
            if pd.notna(timestamp_ist):
                timestamp_ist = timestamp_ist.tz_convert(TIMEZONE)
                timestamp_display = timestamp_ist.strftime("%d-%b-%Y %H:%M:%S IST")

        # Telemetry Status Bar
        render_telemetry_status(
            "LIVE MODE",
            EXPECTED_PROJECT_ID,
            reading.get("document_id", "live_stream"),
            timestamp_display
        )

        # Live Sensor Readings Grid
        st.markdown("### ⚡ Live Electrical Telemetry")
        energy_val = reading.get("energy_kWh")
        energy_str = f"{float(energy_val):.4f}" if energy_val is not None else "0.0000"

        render_sensor_telemetry_grid(
            input_reading["voltage_V"],
            input_reading["current_A"],
            input_reading["power_W"],
            energy_str,
            input_reading["frequency_Hz"],
            input_reading["power_factor"]
        )

        # AI Appliance Identification Showcase
        st.markdown("### 🤖 Neural Appliance Recognition")
        prediction, score = predict_appliance(model, input_reading)
        render_ai_prediction_card(prediction, score, model_choice, input_reading["power_W"])

        # Firebase Diagnostic Information Expander
        with st.expander("🔍 Firebase Connection Diagnostics & Raw Packet", expanded=False):
            st.json(reading)

        # Historical Analytics Graphs
        with st.spinner("Loading telemetry history from Firebase..."):
            historical_df = get_historical_readings(limit=500)

        display_electricity_analytics(historical_df, model=model)

    except Exception as e:
        error_text = str(e)
        if "429" in error_text or "Quota" in error_text:
            st.error("Firestore quota exceeded. Check Firebase usage.")
        else:
            st.error(f"Live dashboard error: {error_text}")


# =====================================================
# 18. MESCOM BILL ESTIMATOR
# =====================================================

display_mescom_bill_estimator()


# =====================================================
# 19. FOOTER
# =====================================================

st.divider()
render_html("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 10px 0;">
    AI-Enabled Smart Electricity Monitoring System • ESP32 • PZEM-004T • LoRa • Wi-Fi • Firebase Firestore • Machine Learning • MESCOM Karnataka
</div>
""")
