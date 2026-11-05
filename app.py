"""Big Data Analytics — tujuh aktivitas Recommendation System."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="RECOMSYS | Big Data Analytics",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="expanded",
)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Semua elemen visual memakai palet Light Analytics. Pengaturan warna widget
# mengikuti .streamlit/config.toml yang disertakan bersama app.py.
st.markdown('''
<style>
:root {
  --page:#F4F7FB; --paper:#FFFFFF; --navy:#172B4D; --ink:#334155;
  --muted:#64748B; --line:#E2E8F0; --blue:#2563EB; --softblue:#EFF5FF;
  --shadow:0 3px 15px rgba(20,47,87,.035);
}
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"] {background:var(--page);color:var(--ink);}
[data-testid="stHeader"] {background:rgba(244,247,251,.97);}
.block-container {max-width:1290px;padding:1.45rem 2rem 3rem;}
/* Stable, consistent spacing across Streamlit's repeated containers. */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {gap:.88rem;}
[data-testid="stVerticalBlock"] {gap:.8rem;}
[data-testid="stHorizontalBlock"] {gap:1rem;align-items:stretch;}
[data-testid="column"] {min-width:0;}
[data-testid="stSidebar"] {background:#FFFFFF;border-right:1px solid var(--line);min-width:288px;max-width:315px;}
[data-testid="stSidebar"] > div:first-child {background:#FFFFFF;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.26rem;}
[data-testid="stSidebar"] .stButton {margin:.04rem 0;}
[data-testid="stSidebar"] .stButton > button {
  display:flex;align-items:center;justify-content:flex-start;width:100%;
  min-height:40px;text-align:left;padding:.57rem .76rem;border:1px solid transparent;
  border-radius:9px;background:transparent;color:#475569;font-weight:500;font-size:.84rem;
  transition:background .15s ease, color .15s ease, border-color .15s ease;
}
[data-testid="stSidebar"] .stButton > button p {font-size:.84rem;line-height:1.3;}
[data-testid="stSidebar"] .stButton > button:hover {background:#F2F6FD;color:#1D4ED8;border-color:#E0E9F8;}
.side-brand {display:flex;align-items:center;gap:11px;padding:9px 2px 14px;}
.brand-icon {width:42px;height:42px;border-radius:12px;background:#EAF1FF;display:flex;justify-content:center;align-items:center;flex-shrink:0;}
.brand-title {color:#152A49;font-size:15px;font-weight:750;line-height:1.25;}
.brand-caption {color:#7A8AA2;font-size:11px;margin-top:3px;}
.side-section {color:#8392AA;font-size:10px;font-weight:800;letter-spacing:.095em;margin:13px 0 6px 6px;}
.side-rule {height:1px;background:#E7EDF4;margin:3px 0 8px;}
.side-footer {border-top:1px solid #E5EBF3;margin-top:19px;padding-top:15px;display:flex;align-items:center;gap:10px;color:#62748F;font-size:11px;}
.side-footer strong {color:#34445C;font-size:12px;}
/* Shared page header, navigation spacing and data summary cards. */
h1,h2,h3,h4 {color:var(--navy);letter-spacing:-.025em;}
h2 {margin-top:.45rem;}
.hero {background:linear-gradient(116deg,#FFFFFF 0%,#FFFFFF 72%,#EFF5FF 100%);
  padding:23px 27px;border:1px solid var(--line);border-radius:17px;
  margin:0 0 10px;box-shadow:var(--shadow);}
.hero h1 {font-size:1.83rem;line-height:1.22;margin:0 0 8px;color:var(--navy);font-weight:780;}
.hero p {font-size:.89rem;line-height:1.55;color:var(--muted);margin:0;max-width:850px;}
.eyebrow {font-size:10px;color:var(--blue);font-weight:800;letter-spacing:.095em;margin-bottom:9px;}
.section-head {margin:15px 0 8px;}
.section-head.tight {margin:15px 0 2px;}
.section-kicker {font-size:10px;color:var(--blue);font-weight:800;letter-spacing:.09em;}
.section-title {font-size:1.27rem;color:var(--navy);font-weight:760;letter-spacing:-.022em;margin:4px 0;line-height:1.25;}
.section-desc {color:var(--muted);font-size:.83rem;line-height:1.53;}
/* Compact and truly balanced metric cards on every analysis page. */
[data-testid="stMetric"] {background:#FFFFFF;border:1px solid var(--line);border-radius:13px;
  min-height:92px;padding:13px 16px;box-shadow:var(--shadow);}
[data-testid="stMetricLabel"] {color:#64748B;font-size:.8rem;}
[data-testid="stMetricValue"] {color:#172B4D;font-weight:760;font-size:1.8rem;line-height:1.17;}
[data-testid="stMetricDelta"] {font-size:.75rem;}
/* Borders use the same token, shadows kept subtle and controls never oversized. */
/* Native st.container(border=True) already supplies a border. Do not add a second one to every nested block. */
[data-testid="stForm"] {background:#FFFFFF;border:1px solid var(--line);border-radius:14px;
  padding:15px 17px;box-shadow:var(--shadow);}
[data-testid="stForm"] [data-testid="stVerticalBlock"] {gap:.65rem;}
[data-testid="stFileUploaderDropzone"] {background:#FCFDFF;border:1px dashed #C8D7EA;border-radius:11px;padding:10px 12px;}
[data-testid="stFileUploader"] label,[data-testid="stWidgetLabel"] p {font-size:.86rem;color:#34445C;font-weight:600;}
.stSelectbox [data-baseweb="select"] > div,.stNumberInput input,.stTextInput input,
.stMultiSelect [data-baseweb="select"] > div {border-radius:9px;background:#FFFFFF;}
/* Separate tab buttons from content without leaving a large empty panel. */
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:5px;border-bottom:1px solid #E1E8F1;padding-bottom:5px;}
[data-testid="stTabs"] [data-baseweb="tab"] {background:transparent;border-radius:9px;padding:7px 13px;
  color:#64748B;font-weight:600;font-size:.86rem;border:1px solid transparent;}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {background:#EAF1FF;color:#1D4ED8;border-color:#DCE8FC;}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {padding-top:12px;}
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:11px;overflow:hidden;background:#FFF;}
[data-testid="stPlotlyChart"] {background:#FFFFFF;border:0;border-radius:13px;padding:4px;box-shadow:none;}
.stDownloadButton button {border:1px solid #C7D8FB;background:#EFF5FF;color:#1D4ED8;
  border-radius:9px;font-size:.83rem;font-weight:650;min-height:35px;}
.stDownloadButton button:hover {background:#E3EEFF;color:#1D4ED8;}
.stButton button[kind="primary"],.stFormSubmitButton button[kind="primary"] {
  background:#2563EB;color:#FFF;border-color:#2563EB;border-radius:9px;font-weight:650;}
.stButton button[kind="primary"]:hover,.stFormSubmitButton button[kind="primary"]:hover {background:#1D4ED8;border-color:#1D4ED8;}
.note {padding:12px 15px;margin:12px 0;background:#EEF5FF;border-left:3px solid var(--blue);
  border-radius:9px;color:#334155;font-size:.84rem;line-height:1.55;}
.smallcap {color:#7C8BA4;font-size:12px;}
/* Overview: cards only as tall as their content; consistent typography and spacing. */
.module-top {display:flex;align-items:center;justify-content:space-between;margin:0 0 10px;}
.module-icon {width:40px;height:40px;border-radius:11px;display:flex;align-items:center;
  justify-content:center;font-size:20px;flex-shrink:0;}
.module-index {font-size:10px;color:#90A0B7;font-weight:750;letter-spacing:.045em;}
.module-title {font-size:1.08rem;font-weight:780;color:#172B4D;margin:0 0 6px;line-height:1.3;}
.module-desc {font-size:.84rem;line-height:1.52;color:#64748B;margin:0 0 12px;min-height:0;}
.module-divider {height:1px;background:#E7ECF4;margin:9px 0;}
.module-label {font-size:.73rem;font-weight:760;color:#334155;margin:0 0 3px;}
.module-study {color:#475569;font-size:.8rem;line-height:1.47;margin:0 0 8px;min-height:0;}
.module-outcome {border-radius:9px;padding:9px 11px;margin:8px 0 7px;}
.module-outcome-label {font-size:9px;color:#64748B;font-weight:750;letter-spacing:.04em;}
.module-outcome-text {font-size:.81rem;font-weight:730;margin-top:3px;line-height:1.4;}
.module-foot {color:#7C8BA4;font-size:.73rem;padding:0;margin:0;}
/* Buttons inside module cards become compact and visually aligned. */
/* Only overview navigation buttons share this style; other page buttons remain native. */
[class*="st-key-overview_"] button {min-height:36px;min-width:0;border-radius:10px;
  font-weight:680;background:#F8FAFF;border:1px solid #D9E4F4;color:#254265;box-shadow:none;}
[class*="st-key-overview_"] button p {font-size:.90rem !important;white-space:nowrap !important;}
[class*="st-key-overview_"] button:hover {border-color:#9CB9F5;background:#EAF1FF;color:#1D4ED8;}
[class*="st-key-overview_3"] button,[class*="st-key-overview_4"] button {background:#F8F4FF;color:#6D28D9;border-color:#E7DBFE;}
[class*="st-key-overview_5"] button,[class*="st-key-overview_6"] button {background:#F0FBF7;color:#047857;border-color:#CDEFE2;}
[class*="st-key-overview_7"] button {background:#FFF7EF;color:#C2410C;border-color:#FEDFC8;}
.workflow-card {background:#FFFFFF;border:1px solid var(--line);border-radius:14px;padding:16px 18px;
  margin:7px 0 12px;box-shadow:var(--shadow);}
.workflow-title {font-size:.95rem;color:#172B4D;font-weight:750;margin:0 0 11px;}
.workflow-line {display:flex;flex-wrap:wrap;align-items:center;gap:7px;}
.workflow-step {background:#EFF5FF;color:#1D4ED8;font-size:.78rem;font-weight:650;
  border:1px solid #DBE9FF;border-radius:8px;padding:8px 10px;}
.workflow-arrow {color:#9FB0CA;font-weight:800;}
@media(max-width:950px) {
 .block-container{padding:1rem 1rem 2rem;}
 .hero{padding:19px 20px;}.hero h1{font-size:1.6rem;}
 [data-testid="stHorizontalBlock"]{gap:.65rem;}
 .workflow-arrow{display:none;}
}

/* V3: gutters and containment: keep *every* panel inside its allocated column. */
*,*::before,*::after {box-sizing:border-box;}
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {min-width:0;max-width:100%;overflow-x:clip;}
.block-container {width:100%;max-width:1210px;padding:2rem clamp(1.3rem,3vw,2.7rem) 3.5rem;}
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"],
[data-testid="column"], [data-testid="stElementContainer"] {min-width:0;max-width:100%;}
/* CSS layout containment prevents long words / chart labels from growing grid columns. */
[data-testid="column"] {overflow-wrap:anywhere;}
[data-testid="stHorizontalBlock"] {gap:1.5rem;}
/* V3 applied padding and borders to *every* internal Streamlit wrapper.
   Leave the native panel alone and target only the intended overview cards. */
[data-testid="stVerticalBlockBorderWrapper"] {min-width:0;max-width:100%;}
[data-testid="stVerticalBlockBorderWrapper"] > div {min-width:0;max-width:100%;}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {min-width:0;}
/* One border per visible module; subtle highlights provide visual interest without extra boxes. */
[data-testid="stVerticalBlockBorderWrapper"]:has(.module-top) {border-radius:16px;}
.module-top + .module-title {margin-top:0;}
/* Forms and metrics keep sufficient breathing room at all screen sizes. */
[data-testid="stMetric"] {padding:20px 24px;min-height:103px;border-radius:15px;max-width:100%;}
[data-testid="stForm"] {padding:23px 25px;border-radius:15px;max-width:100%;overflow:hidden;}
[data-testid="stFileUploaderDropzone"] {padding:18px 19px;max-width:100%;}
.hero {padding:29px 32px;max-width:100%;margin-bottom:15px;}
.note {padding:18px 22px;max-width:100%;}
.section-head {margin:20px 0 12px;}
/* Module cards: more distance from outline, but never fill excessive vertical space. */
.module-top {margin-bottom:14px;}
.module-title {margin:0 0 9px;line-height:1.35;}
.module-desc {margin:0 0 14px;line-height:1.62;}
.module-divider {margin:13px 0;}
.module-study {margin:5px 0 13px;line-height:1.57;}
.module-outcome {margin:12px 0 11px;padding:15px 17px;border-radius:11px;max-width:100%;}
.module-foot {margin-bottom:7px;}
/* Plotly charts and data grids must never extend past their parent card. */
[data-testid="stPlotlyChart"] {
  min-width:0;max-width:100%;width:100%;
  overflow:hidden;padding:18px 20px;border-radius:15px;
}
[data-testid="stPlotlyChart"] > div,
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container,
[data-testid="stPlotlyChart"] .svg-container {
  max-width:100% !important;
}
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  max-width:100%;min-width:0;overflow:hidden;border-radius:12px;
}
[data-testid="stTabs"] {max-width:100%;min-width:0;}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {min-width:0;max-width:100%;padding-top:18px;}
/* Replace an over-wide arrow chain with a five-column flow that truly fits. */
.workflow-card {width:100%;max-width:100%;overflow:hidden;padding:23px 25px;margin:12px 0 22px;}
.workflow-title {margin:0 0 17px;}
.workflow-line {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;align-items:stretch;min-width:0;max-width:100%;}
.workflow-step {position:relative;display:flex;align-items:center;justify-content:center;
 text-align:center;padding:13px 10px;min-width:0;max-width:100%;width:100%;
 font-size:.76rem;overflow-wrap:break-word;white-space:normal;border-radius:9px;line-height:1.4;}
.workflow-step:not(:last-child)::after {content:'›';position:absolute;right:-10px;top:50%;transform:translateY(-50%);
 font-size:21px;font-weight:800;color:#91A8CB;z-index:2;line-height:1;}
.workflow-arrow {display:none;}
@media(max-width:1150px) {
 .workflow-line {grid-template-columns:repeat(3,minmax(0,1fr));}
 .workflow-step::after {display:none;}
}
@media(max-width:700px) {
 .block-container {padding:1.15rem 1.05rem 2rem;}
 .hero {padding:21px 22px;}
 [data-testid="stHorizontalBlock"] {gap:.9rem;}
 /* Keep native container padding on mobile too. */
 [data-testid="stMetric"] {padding:15px 17px;}
 [data-testid="stForm"] {padding:18px;}
 [data-testid="stPlotlyChart"] {padding:10px;}
 .workflow-card {padding:18px;}
 .workflow-line {grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;}
}
@media(max-width:420px) {.workflow-line {grid-template-columns:1fr;}}


/* V4 — Typography & accent polish; keep card sizes and page gutters unchanged. */
.hero h1 {font-size:2.08rem;line-height:1.2;}
.hero p {font-size:1rem;line-height:1.63;}
.eyebrow,.section-kicker {font-size:11px;}
.section-title {font-size:1.52rem;line-height:1.25;}
.section-desc {font-size:.96rem;line-height:1.6;}
.section-head {margin-top:20px;}
.module-top {margin-bottom:15px;}
.module-icon {width:44px;height:44px;font-size:22px;}
.module-index {font-size:11px;}
.module-title {font-size:1.24rem;line-height:1.3;margin-bottom:9px;}
.module-desc {font-size:.97rem;line-height:1.58;color:#465A74;}
.module-label {font-size:.84rem;}
.module-study {font-size:.94rem;line-height:1.55;color:#465A74;}
.module-outcome {margin:12px 0 11px;padding:15px 17px;}
.module-outcome-label {font-size:10px;}
.module-outcome-text {font-size:.96rem;line-height:1.5;}
.module-foot {font-size:.84rem;color:#687B95;}
.module-index {padding:6px 10px;border:1px solid #E1E9F4;border-radius:999px;background:#F7FAFF;}
[data-testid="stMetricLabel"] p {font-size:.92rem !important;color:#4D607A !important;}
[data-testid="stMetricValue"] {font-size:2.03rem;}
[data-testid="stWidgetLabel"] p, [data-testid="stFileUploader"] label {font-size:.96rem !important;}
[data-testid="stTabs"] [data-baseweb="tab"] {font-size:.94rem;}
[data-testid="stDataFrame"] {font-size:.9rem;}
.note {font-size:.95rem;line-height:1.65;}
.workflow-title {font-size:1.08rem;}
.workflow-step {font-size:.88rem;}
/* A little depth from color, not extra frames. */
.hero {background:linear-gradient(120deg,#FFFFFF 0%,#F3F7FF 78%,#EAF5FF 100%);}
[data-testid="stMetric"] {background:linear-gradient(145deg,#FFFFFF 72%,#EFF5FF 100%);border-color:#DCE7F6;}
[data-testid="stMetric"]:nth-of-type(even) {background:linear-gradient(145deg,#FFFFFF 70%,#F2EEFF 100%);}
/* Preserve readable font sizing on medium screens. */
@media(max-width:900px) {
 .hero h1{font-size:1.75rem;}
 .module-title{font-size:1.1rem;}
 .module-desc,.module-study{font-size:.9rem;}
 [class*="st-key-overview_"] button p{font-size:.82rem !important;}
}


/* V5 — Visual refresh: modern, spacious analytics UI.
   Layout-only styles: no analytical logic, data loader, or module callbacks changed. */
:root {
 --page:#F5F7FC;--paper:#FFFFFF;--navy:#142342;--ink:#34445E;
 --muted:#72809A;--line:#E2E9F3;--blue:#3563E9;--softblue:#EDF3FF;
 --shadow:0 12px 32px rgba(25,50,99,.055);
}
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"] {
 background:radial-gradient(circle at 80% 5%,#EEF3FF 0,transparent 34%),var(--page);
}
[data-testid="stHeader"]{background:rgba(245,247,252,.88);backdrop-filter:blur(12px);}
.block-container{max-width:1270px;padding:2rem clamp(1.3rem,3.1vw,3rem) 3.7rem;}

/* Sidebar navigation keeps real gutters between active menu and both edges. */
[data-testid="stSidebar"] {
 min-width:331px;max-width:331px;
 background:#FCFDFF;border-right:1px solid #E0E7F4;
 box-shadow:6px 0 26px rgba(17,36,75,.025);
}
[data-testid="stSidebar"] > div:first-child {background:#FCFDFF;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.30rem;}
[data-testid="stSidebar"] [class*="st-key-nav_"]{
 width:calc(100% - 18px)!important;margin-left:9px!important;
 margin-right:9px!important;margin-bottom:3px!important;
}
[data-testid="stSidebar"] .stButton > button{
 min-height:44px;border-radius:11px;padding:.64rem .85rem!important;
 color:#536681;gap:.62rem;font-weight:570;font-size:.84rem;
 transition:background .16s ease,border-color .16s ease,box-shadow .16s ease;
}
[data-testid="stSidebar"] .stButton > button p{
 font-size:.82rem;letter-spacing:-.006em;line-height:1.38;
}
[data-testid="stSidebar"] .stButton > button:hover{
 background:#EFF4FF;color:#2854CD;border-color:#E1EAFB;
 box-shadow:0 3px 10px rgba(53,99,233,.055);
}
.side-brand{margin:6px 7px 1px;padding:12px 4px 16px;gap:13px;}
.brand-icon{width:46px;height:46px;border-radius:15px;
 background:linear-gradient(145deg,#DCE9FF 0%,#ECF4FF 100%);
 box-shadow:inset 0 0 0 1px #D2E2FF;}
.brand-title{font-size:15px;font-weight:790;letter-spacing:-.02em;color:#182B51;}
.brand-caption{font-size:11px;color:#8291A9;}
.side-rule{background:linear-gradient(90deg,#E6ECF5,#CFDDF2,#E6ECF5);margin:5px 8px 9px;}
.side-section{color:#8B9BB5;font-size:10px;font-weight:800;
 letter-spacing:.105em;margin:16px 0 7px 17px;}
.side-footer{
 margin:22px 9px 10px;padding:16px 12px 12px;
 border-top:1px solid #E1E9F5;border-radius:0;color:#8291A6;
}
.side-footer strong{color:#253E67;}

/* Editorial-style headers with a high-contrast visual focal point. */
.hero{
 position:relative;isolation:isolate;overflow:hidden;
 background:linear-gradient(110deg,#132544 2%,#1A3767 55%,#2755A2 100%);
 border:1px solid rgba(107,154,242,.24);border-radius:22px;
 padding:34px 38px;box-shadow:0 18px 34px rgba(23,47,91,.16);
 margin-bottom:23px;
}
.hero::after{content:'';position:absolute;z-index:-1;width:300px;height:300px;
 border-radius:50%;right:-110px;top:-155px;background:rgba(109,157,255,.16);}
.hero h1{color:#FFFFFF!important;font-size:2.13rem;line-height:1.16;margin-bottom:11px;}
.hero p{color:#CAD8EF;font-size:1rem;max-width:760px;line-height:1.68;}
.hero .eyebrow{color:#A9C6FF;font-size:11px;letter-spacing:.115em;margin-bottom:12px;}
.hero-overview{padding:43px 44px 40px;min-height:300px;display:flex;align-items:center;
 justify-content:space-between;gap:22px;background:linear-gradient(117deg,#101F3B 0%,#173365 55%,#2B58AE 100%);}
.hero-overview::before{content:'';position:absolute;z-index:-1;inset:0;
 background:radial-gradient(circle at 80% 60%,rgba(117,159,255,.19),transparent 35%);}
.hero-overview-copy{position:relative;max-width:620px;z-index:2;}
.hero-overview h1{font-size:clamp(2.25rem,3.55vw,3.1rem);line-height:1.1;
 letter-spacing:-.045em;max-width:550px;margin:14px 0;}
.hero-overview h1 span{color:#A6C4FF;}
.hero-overview p{font-size:.99rem;max-width:570px;line-height:1.67;margin:0;}
.hero-pill{display:inline-flex;align-items:center;gap:8px;border-radius:999px;
 padding:7px 12px;background:rgba(153,188,255,.12);border:1px solid rgba(189,211,255,.24);
 color:#DAE8FF;font-weight:730;font-size:.69rem;letter-spacing:.08em;}
.hero-dot{width:6px;height:6px;background:#72E8C3;border-radius:50%;box-shadow:0 0 0 4px rgba(114,232,195,.12);}
.hero-overview-foot{margin-top:21px;display:flex;gap:10px;flex-wrap:wrap;}
.hero-overview-foot span{font-size:.76rem;color:#DFE9FB;border:1px solid rgba(180,206,255,.18);
 padding:7px 10px;border-radius:8px;background:rgba(255,255,255,.045);}
.hero-illustration{width:270px;height:235px;flex:none;position:relative;display:flex;align-items:center;justify-content:center;}
.hero-orbit{width:218px;height:218px;border:1px solid rgba(167,196,255,.24);
 border-radius:50%;display:flex;align-items:center;justify-content:center;position:relative;}
.hero-orbit::before{content:'';position:absolute;inset:22px;border:1px dashed rgba(194,215,255,.26);border-radius:50%;}
.hero-orbit::after{content:'';position:absolute;inset:46px;border:1px solid rgba(210,226,255,.17);border-radius:50%;}
.hero-core{height:91px;width:91px;border-radius:25px;
 background:linear-gradient(148deg,#7D9FFF,#4B75E6);display:flex;align-items:center;justify-content:center;
 box-shadow:0 12px 30px rgba(11,27,71,.25),inset 0 1px 0 rgba(255,255,255,.37);z-index:1;}
.hero-satellite{position:absolute;width:39px;height:39px;border-radius:12px;
 display:flex;align-items:center;justify-content:center;color:white;
 border:1px solid rgba(219,233,255,.22);box-shadow:0 9px 18px rgba(9,29,65,.21);}
.hero-satellite.s1{top:12px;left:28px;background:#436FE6;}
.hero-satellite.s2{right:1px;top:75px;background:#20A7AB;}
.hero-satellite.s3{bottom:15px;left:34px;background:#7C60D4;}
.hero-satellite svg,.hero-core svg{display:block;}

/* Clear information hierarchy and less generic native card styling. */
.section-head{margin:27px 0 14px;}
.section-kicker{font-size:11px;color:#4168C8;font-weight:800;letter-spacing:.09em;}
.section-title{font-size:1.65rem;color:#182B51;font-weight:790;letter-spacing:-.03em;}
.section-desc{color:#6D7F98;font-size:.96rem;max-width:820px;}
[data-testid="stMetric"]{
 background:#FFF;border:1px solid #E0E8F4;border-radius:16px;
 box-shadow:0 7px 22px rgba(27,53,99,.045);padding:22px 25px;min-height:115px;
}
[data-testid="stMetric"]:nth-of-type(even){background:#FFF;}
[data-testid="stMetricLabel"] p{font-size:.9rem!important;color:#71819B!important;}
[data-testid="stMetricValue"]{font-size:2.12rem;color:#1D355E;}
[data-testid="stVerticalBlockBorderWrapper"]:has(.module-top){
 border-radius:19px;box-shadow:0 10px 27px rgba(24,49,102,.065);
 background:#FFF;transition:box-shadow .16s ease,transform .16s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.module-top):hover{
 box-shadow:0 16px 35px rgba(24,49,102,.105);transform:translateY(-2px);
}
.module-top{margin-bottom:16px;}
.module-icon{width:49px;height:49px;border-radius:14px;font-size:23px;}
.module-index{font-size:10px;background:#F6F8FC;border-color:#E6EDF7;color:#7D8FA9;}
.module-title{font-size:1.3rem;font-weight:790;letter-spacing:-.025em;margin:0 0 9px;}
.module-desc{font-size:.96rem;line-height:1.66;color:#63738B;margin-bottom:18px;min-height:5rem;}
.module-divider{background:#E9EFF7;margin:15px 0;}
.module-label{color:#334969;font-size:.83rem;}
.module-study{font-size:.92rem;color:#566B88;line-height:1.57;margin:5px 0 15px;}
.module-outcome{border-radius:13px;margin:10px 0 13px;padding:15px 18px;}
.module-outcome-label{font-size:.64rem;letter-spacing:.08em;}
.module-outcome-text{font-size:.93rem;}
.module-foot{font-size:.77rem;color:#7D8DA7;}
[class*="st-key-overview_"] button{
 min-height:41px;border-radius:10px;font-weight:730;
 transition:transform .14s ease,box-shadow .14s ease,border-color .14s ease;
}
[class*="st-key-overview_"] button:hover{transform:translateY(-1px);box-shadow:0 7px 14px rgba(42,77,138,.08);}
.workflow-card{border-radius:18px;padding:24px 27px;box-shadow:0 8px 22px rgba(27,53,99,.05);}
.workflow-step{border-radius:11px;background:#F1F5FD;border-color:#DBE5F6;color:#37558D;}
[data-testid="stForm"]{border-radius:17px;box-shadow:0 7px 20px rgba(26,49,91,.045);}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
 background:#E8EFFF;color:#2655CE;border:1px solid #C7D9FD;
}
[data-testid="stPlotlyChart"]{border-radius:15px;}
@media(max-width:1100px){
 .hero-overview{padding:32px 32px;min-height:250px;}
 .hero-illustration{width:190px;height:190px;transform:scale(.78);transform-origin:center;}
}
@media(max-width:750px){
 [data-testid="stSidebar"]{min-width:292px;max-width:315px;}
 .block-container{padding:1.2rem 1.05rem 2.2rem;}
 .hero,.hero-overview{padding:27px 24px;border-radius:17px;}
 .hero-overview{min-height:unset;}
 .hero-overview h1{font-size:2.18rem;}
 .hero-illustration{display:none;}
 .hero h1{font-size:1.79rem;}
 .section-title{font-size:1.36rem;}
}


/* Overview only: one consistent inset for metrics, sections, cards and dataset.
   IMPORTANT: use a plain container, not an extra st.columns wrapper. */
.st-key-overview_content {
  width: calc(100% - 48px);
  max-width: calc(100% - 48px);
  margin-left: auto;
  margin-right: auto;
  min-width: 0;
  padding-top: 24px !important;
}
@media (max-width: 750px) {
  .st-key-overview_content {
    width: calc(100% - 24px);
    max-width: calc(100% - 24px);
  }
}




/* Overview V6: Learning Journey */
.journey-shell {
  background: linear-gradient(126deg,#FFFFFF 0%,#FBFCFF 52%,#F2F6FF 100%);
  border: 1px solid #DBE5F4;
  border-radius: 22px;
  padding: 27px 28px 22px;
  box-shadow: 0 9px 25px rgba(27,57,111,.06);
  overflow: hidden;
  box-sizing: border-box !important;
  width: auto !important;
  max-width: none !important;
  margin: 15px 48px 30px 0 !important;

}
.journey-head {display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:28px;flex-wrap:wrap;}
.journey-heading {font-size:1.12rem;font-weight:800;color:#1B3157;letter-spacing:-.025em;}
.journey-subheading {color:#74859F;font-size:.85rem;margin-top:4px;}
.journey-pill {font-size:.68rem;letter-spacing:.085em;font-weight:800;color:#3363D3;
  padding:7px 11px;background:#EDF3FF;border:1px solid #DCE7FB;border-radius:999px;}
.journey-grid {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;position:relative;}
.journey-grid::before {content:'';position:absolute;z-index:0;left:10%;right:10%;top:36px;height:3px;
  background:linear-gradient(90deg,#3971EF 0%,#8367EA 30%,#00A8A3 66%,#0EAE83 100%);
  border-radius:4px;opacity:.33;}
.journey-step {min-width:0;position:relative;z-index:1;text-align:center;padding:0 5px;
  transition:transform .18s ease;}
.journey-step:hover {transform:translateY(-4px);}
.journey-step:nth-child(1){--step-a:#2563EB;--step-b:#5E8AFF;}
.journey-step:nth-child(2){--step-a:#6D4BD9;--step-b:#9B79F1;}
.journey-step:nth-child(3){--step-a:#0D93B8;--step-b:#49C0D4;}
.journey-step:nth-child(4){--step-a:#0F9A86;--step-b:#4BCDB2;}
.journey-step:nth-child(5){--step-a:#E7863A;--step-b:#F6B060;}
.journey-node {display:flex;align-items:center;justify-content:center;margin:0 auto 14px;
  width:74px;height:74px;border-radius:22px;color:white;position:relative;
  border:5px solid #FFFFFF;background:linear-gradient(145deg,var(--step-a),var(--step-b));
  box-shadow:0 8px 22px rgba(25,66,123,.18);}
.journey-node svg {width:27px;height:27px;fill:none;stroke:currentColor;stroke-width:1.8;
  stroke-linecap:round;stroke-linejoin:round;}
.journey-num {color:var(--step-a);font-size:.66rem;font-weight:850;letter-spacing:.12em;margin-bottom:7px;}
.journey-name {font-weight:790;color:#1B3157;font-size:.91rem;line-height:1.35;margin-bottom:7px;}
.journey-description {font-size:.77rem;line-height:1.5;color:#6D809B;max-width:195px;margin:0 auto;}
.journey-footer {display:flex;align-items:center;gap:9px;margin-top:23px;padding-top:17px;
  border-top:1px dashed #DFE7F4;color:#647894;font-size:.8rem;line-height:1.5;}
.journey-footer-dot {height:8px;width:8px;flex:none;border-radius:100%;background:#34B39D;
  box-shadow:0 0 0 4px #DDF8F1;}
@media (max-width:850px) {
  .journey-shell{padding:22px 20px;}
  .journey-head{margin-bottom:22px;}
  .journey-grid {display:flex;flex-direction:column;gap:18px;}
  .journey-grid::before {top:29px;bottom:29px;left:28px;right:auto;width:3px;height:auto;}
  .journey-step {display:flex;align-items:center;gap:14px;padding:0;text-align:left;}
  .journey-node {flex:none;width:58px;height:58px;margin:0;border-radius:18px;border-width:4px;}
  .journey-node svg {width:22px;height:22px;}
  .journey-name {margin-bottom:3px;}
  .journey-num {margin-bottom:4px;}
  .journey-description {max-width:none;margin:0;}
  .journey-footer {margin-top:20px;}
}

</style>
''', unsafe_allow_html=True)

NAV = [
    "Overview",
    "Apriori Analysis",
    "Market Basket Analysis",
    "Content-Based Recommendation",
    "Movie Recommendation",
    "User-Based Collaborative Filtering",
    "Item-Based Collaborative Filtering",
    "Hybrid Recommendation",
]
NAV_KEYS = ["overview", "apriori", "market", "content", "movie", "user", "item", "hybrid"]
NAV_ICONS = [
    ":material/dashboard:", ":material/shopping_cart:",
    ":material/shopping_basket:", ":material/description:",
    ":material/movie:", ":material/groups:",
    ":material/menu_book:", ":material/call_merge:",
]
NAV_GROUPS = [
    ("WORKSPACE", [0]),
    ("01 / ASSOCIATION RULES", [1, 2]),
    ("02 / CONTENT-BASED FILTERING", [3, 4]),
    ("03 / COLLABORATIVE FILTERING", [5, 6]),
    ("04 / HYBRID SYSTEM", [7]),
]

def select_nav(index):
    st.session_state["active_page"] = NAV[int(index)]

if st.session_state.get("active_page") not in NAV:
    st.session_state["active_page"] = NAV[0]

with st.sidebar:
    st.markdown('''
    <div class="side-brand">
      <div class="brand-icon">
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <rect x="5" y="5" width="18" height="18" rx="6" stroke="#2563EB" stroke-width="1.8"/>
          <circle cx="10" cy="11" r="2" fill="#2563EB"/>
          <circle cx="18" cy="10" r="2" fill="#2563EB"/>
          <circle cx="16" cy="18" r="2" fill="#2563EB"/>
          <path d="M11.8 11L16.2 10.2M11.3 12.7L15 16.4M17.9 12L16.6 16" stroke="#2563EB" stroke-width="1.4" stroke-linecap="round"/>
        </svg>
      </div>
      <div><div class="brand-title">Big Data Analytics</div>
      <div class="brand-caption">Created by Kelompok 1 (Amila dan Ziharul)</div></div>
    </div><div class="side-rule"></div>''', unsafe_allow_html=True)
    for group_label, positions in NAV_GROUPS:
        st.markdown(f'<div class="side-section">{group_label}</div>', unsafe_allow_html=True)
        for i in positions:
            st.button(
                NAV[i], key="nav_" + NAV_KEYS[i],
                icon=NAV_ICONS[i], use_container_width=True,
                on_click=select_nav, args=(i,),
            )
    st.markdown('''<div class="side-footer">
      <svg width="25" height="25" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="m2 10 10-5 10 5-10 5z"/><path d="M6 12v5c3.6 2.6 8.4 2.6 12 0v-5"/><path d="M22 10v6"/>
      </svg>
      <div><strong>MMT ITS</strong><div>Academic Dashboard · 2026</div></div>
    </div>''', unsafe_allow_html=True)
    # Tandai item aktif tanpa menggunakan radio button atau navigasi JavaScript.
    selected_key = NAV_KEYS[NAV.index(st.session_state["active_page"])]
    st.markdown(f'''<style>
    [data-testid="stSidebar"] .st-key-nav_{selected_key} button {{
        background:linear-gradient(105deg,#E5EEFF,#F0F5FF) !important;
        color:#2353CC !important;border:1px solid #CCDDFD !important;
        font-weight:750 !important;
        box-shadow:inset 3px 0 0 #3563E9,0 5px 12px rgba(34,83,181,.08);
    }}
    [data-testid="stSidebar"] .st-key-nav_{selected_key} button p {{color:#1D4ED8 !important;font-weight:700 !important;}}
    </style>''', unsafe_allow_html=True)

page = st.session_state["active_page"]


def hero(tag, title, detail):
    st.markdown(f'<div class="hero"><div class="eyebrow">{tag}</div>'
                f'<h1>{title}</h1><p>{detail}</p></div>', unsafe_allow_html=True)


def note(message):
    st.markdown(f'<div class="note">{message}</div>', unsafe_allow_html=True)


def metrics(entries):
    cols = st.columns(len(entries))
    for col, (label, value) in zip(cols, entries):
        col.metric(label, value)


def show_table(df, filename="hasil_analisis.csv", height=320):
    compact_dataframe(df, height=height, hide_index=True)
    st.download_button("Unduh hasil CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=filename, mime="text/csv", icon=":material/download:")


def compact_dataframe(df, height=None, **kwargs):
    """Show results without reserving hundreds of empty pixels for small datasets."""
    try:
        rows = len(df)
    except (ValueError, TypeError):
        rows = 10
    ideal = max(115, min(400, 39 * (min(rows, 10) + 1)))
    actual = min(ideal, height) if height else ideal
    kwargs.setdefault("use_container_width", True)
    st.dataframe(df, height=actual, **kwargs)


def section_heading(kicker, heading, description=""):
    """A single typographic hierarchy for all analysis activities."""
    st.markdown(
        f'<div class="section-head tight"><div class="section-kicker">{kicker}</div>'
        f'<div class="section-title">{heading}</div>'
        f'<div class="section-desc">{description}</div></div>',
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=360):
    chart_accent = {
        "Apriori Analysis":"#2563EB", "Market Basket Analysis":"#0F9D8A",
        "Content-Based Recommendation":"#7C3AED", "Movie Recommendation":"#7C3AED",
        "User-Based Collaborative Filtering":"#08916B",
        "Item-Based Collaborative Filtering":"#08916B", "Hybrid Recommendation":"#EA580C",
    }.get(page, "#2563EB")
    fig.update_layout(
        template="plotly_white", height=height,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        margin=dict(l=15, r=15, t=45, b=25),
        font=dict(color="#334155", size=14),
        colorway=[chart_accent, "#7C3AED", "#059669", "#EA580C"],
        title_font=dict(color="#172B4D", size=17),
        autosize=True,
        xaxis=dict(gridcolor="#E9EEF6", zerolinecolor="#E9EEF6", automargin=True),
        yaxis=dict(gridcolor="#E9EEF6", zerolinecolor="#E9EEF6", automargin=True),
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    # Only recolor plain bar traces; continuous-color visualizations keep their meaningful scale.
    for trace in fig.data:
        if getattr(trace, "type", None) == "bar" and getattr(trace, "marker", None) is not None:
            current = getattr(trace.marker, "color", None)
            if current is None or (isinstance(current, str) and current.lower() == "#636efa"):
                trace.marker.color = chart_accent
    st.plotly_chart(fig, use_container_width=True, config={"responsive":True,"displaylogo":False})

@st.cache_data(show_spinner=False)
def groceries_raw():
    return pd.read_csv(DATA / "groceries.csv", header=None, dtype=str)


@st.cache_data(show_spinner=False)
def movies_raw():
    return pd.read_csv(DATA / "movie.csv")


@st.cache_data(show_spinner=False)
def ratings_raw():
    return pd.read_csv(DATA / "rating.csv")


@st.cache_data(show_spinner=False)
def book_interactions():
    return pd.read_csv(DATA / "book_knn_interactions.csv.gz", compression="gzip")


def transactions_from_frame(frame):
    return [list(dict.fromkeys(str(x).strip() for x in row.dropna().tolist() if str(x).strip() and str(x).strip().lower() != 'nan')) for _,row in frame.iterrows() if row.notna().any()]


def association_rules_inclusive(itemsets, minimum_confidence):
    """Batas confidence bersifat inklusif, termasuk nilai tepat di ambang.

    Misalnya 0.6 / 0.8 di floating-point menjadi 0.7499999999999999,
    sehingga 75% bisa terbuang jika langsung dibandingkan dengan 0.75.
    Toleransi numerik kecil ini mengatasi masalah tersebut.
    """
    threshold = max(0.0, float(minimum_confidence) - 1e-12)
    return association_rules(itemsets, metric="confidence", min_threshold=threshold)


def apriori_tables(transactions, minimum_support, minimum_confidence, maximum_length=None):
    encoder=TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    basket = pd.DataFrame(encoded, columns=encoder.columns_)
    itemsets = apriori(basket, min_support=minimum_support, use_colnames=True, max_len=maximum_length)
    if len(itemsets):
        rules = association_rules_inclusive(itemsets, minimum_confidence)
    else:
        rules = pd.DataFrame()
    return itemsets, rules, basket


@st.cache_data(show_spinner="Menghitung Apriori dan aturan asosiasi...")
def run_groceries(support, confidence, max_len, uploaded_bytes=None):
    frame = pd.read_csv(BytesIO(uploaded_bytes), header=None, dtype=str) if uploaded_bytes else groceries_raw()
    transactions = transactions_from_frame(frame)
    # Pengaman Streamlit Cloud: hindari Apriori tanpa batas panjang itemset.
    # low_memory=True menurunkan konsumsi memori saat membangkitkan kandidat.
    if max_len not in (2, 3):
        raise ValueError("Panjang kombinasi hanya boleh 2 atau 3 untuk dashboard cloud.")
    if len(transactions) > 50000:
        raise ValueError("Dataset terlalu besar untuk paket cloud ini (maksimal 50.000 transaksi).")
    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    basket = pd.DataFrame(encoded, columns=encoder.columns_)
    if basket.shape[1] > 300:
        raise ValueError("Dataset memiliki lebih dari 300 jenis barang. Saring dahulu item yang jarang muncul.")
    itemsets = apriori(basket, min_support=support, use_colnames=True,
                      max_len=max_len, low_memory=True)
    rules = association_rules_inclusive(itemsets, confidence) if len(itemsets) else pd.DataFrame()
    freq = basket.sum().sort_values(ascending=False)
    return itemsets, rules, freq, len(transactions), basket.shape[1]


def pretty_itemsets(frame):
    d=frame.copy()
    if 'itemsets' in d.columns:
        d['itemsets'] = d['itemsets'].apply(lambda s: ', '.join(sorted(s)))
    return d


def pretty_rules(frame):
    if frame.empty:
        return pd.DataFrame(columns=['Jika membeli', 'Maka membeli', 'Support', 'Confidence', 'Lift'])
    d=frame.copy()
    d['Jika membeli']=d['antecedents'].apply(lambda s:', '.join(sorted(s)))
    d['Maka membeli']=d['consequents'].apply(lambda s:', '.join(sorted(s)))
    return d[['Jika membeli','Maka membeli','support','confidence','lift']].rename(columns={'support':'Support','confidence':'Confidence','lift':'Lift'}).sort_values(['Lift','Confidence'],ascending=False).reset_index(drop=True)


EXAMPLE_BASKET = [
    ['Roti', 'Susu', 'Mentega'], ['Roti', 'Mentega'],
    ['Roti', 'Susu', 'Selai'], ['Apel', 'Pisang'],
    ['Roti', 'Susu', 'Mentega', 'Selai']
]

EXAMPLE_CONTENT = pd.DataFrame({
    'Judul':['The Matrix','Inception','Interstellar','The Notebook','Titanic'],
    'Deskripsi':[
        'Action sci-fi hacker virtual reality computer illusion',
        'Action sci-fi dream mind control heist illusion',
        'Sci-fi space travel black hole time relativity',
        'Romance love story drama beautiful couple memory',
        'Romance drama ship tragedy beautiful love ocean']
})

EXAMPLE_RATINGS = pd.DataFrame({
    'Pengguna':['Andi','Andi','Andi','Budi','Budi','Citra','Citra','Citra','Citra'],
    'Film':['The Matrix','Inception','Avatar','The Matrix','Inception','The Matrix','Inception','Avatar','Titanic'],
    'Rating':[5,4,1,4,5,4,5,1,5]
})
EXAMPLE_HYBRID = pd.DataFrame({
    'Movie':['The Matrix','Inception','Avatar','Titanic','The Notebook'],
    'Genre':['Action Sci-Fi','Action Sci-Fi','Sci-Fi Adventure','Romance Drama','Romance Drama']
})


def ratings_matrix(frame):
    matrix=frame.pivot_table(index='Pengguna', columns='Film',values='Rating').fillna(0)
    sim=pd.DataFrame(cosine_similarity(matrix), index=matrix.index,columns=matrix.index)
    return matrix,sim


@st.cache_data(show_spinner=False)
def prep_movies():
    movies=movies_raw().copy()
    movies['title_year']=movies['title'].str.strip()
    movies['year']=pd.to_numeric(movies['title_year'].str.extract(r'\((\d{4})\)\s*$')[0],errors='coerce')
    movies['title_clean']=movies['title_year'].str.replace(r'\s*\(\d{4}\)\s*$', '',regex=True)
    movies=movies[movies.genres.fillna('(no genres listed)')!='(no genres listed)'].copy().reset_index(drop=True)
    genre_text=movies.genres.str.replace('|',' ',regex=False).str.replace('Sci-Fi','Scifi',regex=False).str.replace('Film-Noir','Noir',regex=False)
    vector=TfidfVectorizer(stop_words='english')
    matrix=vector.fit_transform(genre_text)
    return movies,matrix


@st.cache_resource(show_spinner="Menyiapkan matriks buku dan model kNN...")
def prep_books():
    books=book_interactions()
    pivot = books.pivot_table(index='bookTitle',columns='userID',values='bookRating',aggfunc='mean',fill_value=0)
    sparse=csr_matrix(pivot.to_numpy(dtype=np.float32))
    knn=NearestNeighbors(metric='cosine',algorithm='brute')
    knn.fit(sparse)
    return pivot,sparse,knn


def render_overview():
    st.markdown('''<div class="hero hero-overview">
      <div class="hero-overview-copy">
        <div class="hero-pill"><span class="hero-dot"></span> BIG DATA ANALYTICS · MMT ITS</div>
        <h1>Build insights with<br><span>Recommendation Systems</span></h1>
        <p>Laboratorium pembelajaran interaktif untuk memahami metode sistem rekomendasi,
        mengeksplorasi data, dan membandingkan hasil analisis melalui tujuh aktivitas praktik.</p>
        <div class="hero-overview-foot">
          <span>Association Rules</span><span>Content-Based</span>
          <span>Collaborative</span><span>Hybrid</span>
        </div>
      </div>
      <div class="hero-illustration" aria-hidden="true">
        <div class="hero-orbit">
          <div class="hero-core">
            <svg width="51" height="51" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M13 15L25 10L39 19L33 36L17 40L13 15Z" stroke="white" stroke-width="2.4" stroke-linejoin="round" opacity=".8"/>
              <path d="M13 15L28 25L39 19M28 25L33 36L17 40L13 15" stroke="white" stroke-width="1.7" opacity=".65"/>
              <circle cx="13" cy="15" r="4" fill="white"/><circle cx="25" cy="10" r="4" fill="white"/>
              <circle cx="39" cy="19" r="4" fill="white"/><circle cx="28" cy="25" r="5" fill="#B6F3E8"/>
              <circle cx="33" cy="36" r="4" fill="white"/><circle cx="17" cy="40" r="4" fill="white"/>
            </svg>
          </div>
        </div>
        <div class="hero-satellite s1"><svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8" stroke-linecap="round"><path d="M4 19V10M10 19V5M16 19v-8M22 19V8"/></svg></div>
        <div class="hero-satellite s2"><svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8" stroke-linecap="round"><circle cx="6" cy="12" r="2"/><circle cx="18" cy="6" r="2"/><circle cx="18" cy="18" r="2"/><path d="M8 11l8-4M8 13l8 4"/></svg></div>
        <div class="hero-satellite s3"><svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8" stroke-linecap="round"><path d="M4 13l5-5 4 4 7-7M16 5h4v4"/></svg></div>
      </div>
    </div>''', unsafe_allow_html=True)
    # Gutter konten tanpa kolom pembungkus: hindari nested-column API exception.
    # Hanya Overview, sehingga layout tujuh aktivitas lain tidak berubah.
    with st.container(key="overview_content", border=False):
        metrics([("Modul Pembelajaran", "4"), ("Aktivitas Praktik", "7")])
        st.markdown('''<div class="section-head">
          <div class="section-kicker">EXPLORE OUR MODULES</div>
          <div class="section-title">Learning Modules</div>
          <div class="section-desc">Empat pendekatan utama untuk memahami cara kerja sistem rekomendasi.
          Setiap modul menjelaskan tujuan pembelajaran, konsep yang digunakan, dan keluaran analisis.</div>
        </div>''', unsafe_allow_html=True)

        modules = [
            dict(
                no="01", name="Association Rule Mining", mark="◆", color="#2563EB", bg="#EAF1FF",
                outcome_bg="#EFF5FF", desc="Mempelajari cara menemukan keterkaitan antarbarang "
                "dari riwayat transaksi, termasuk barang yang sering dibeli bersamaan.",
                lessons="Apriori, frequent itemsets, support, confidence, dan lift.",
                outcome="Pola pembelian dan aturan asosiasi", foot="2 aktivitas · Apriori & Market Basket",
                actions=[(1,"Apriori"),(2,"Market Basket")],
            ),
            dict(
                no="02", name="Content-Based Filtering", mark="▣", color="#7C3AED", bg="#F1ECFF",
                outcome_bg="#F7F3FF", desc="Mempelajari bagaimana kesamaan isi dan karakteristik "
                "item digunakan untuk merekomendasikan film yang memiliki ciri serupa.",
                lessons="TF-IDF, cosine similarity, dan pemeringkatan kemiripan konten.",
                outcome="Daftar rekomendasi film serupa", foot="2 aktivitas · Content-Based & Movie",
                actions=[(3,"Content-Based"),(4,"Movie")],
            ),
            dict(
                no="03", name="Collaborative Filtering", mark="◉", color="#059669", bg="#E7F8F0",
                outcome_bg="#EEFAF4", desc="Mempelajari rekomendasi berdasarkan kesamaan "
                "preferensi pengguna dan pola penilaian antaritem.",
                lessons="User-Based Filtering serta Item-Based Filtering dengan kNN.",
                outcome="Rekomendasi dari pola rating", foot="2 aktivitas · User-Based & Item-Based",
                actions=[(5,"User-Based"),(6,"Item-Based")],
            ),
            dict(
                no="04", name="Hybrid Recommendation", mark="◇", color="#EA580C", bg="#FFF0E5",
                outcome_bg="#FFF7ED", desc="Mempelajari penggabungan rekomendasi "
                "berbasis kesamaan konten dan preferensi pengguna agar keputusan memanfaatkan beberapa sumber informasi.",
                lessons="Kombinasi skor Content-Based dan Collaborative, normalisasi, serta pembobotan.",
                outcome="Rekomendasi gabungan dan pemeringkatan", foot="1 aktivitas · Hybrid Recommendation",
                actions=[(7,"Hybrid Recommendation")],
            ),
        ]

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        for start in range(0, len(modules), 2):
            columns = st.columns(2, gap="large")
            for col, module in zip(columns, modules[start:start+2]):
                with col:
                    with st.container(border=True):
                        st.markdown(f'''
                        <div class="module-top" style="border-top:3px solid {module['color']};padding-top:17px">
                          <div class="module-icon" style="background:{module['bg']};color:{module['color']}">{module['mark']}</div>
                          <div class="module-index">MODULE {module['no']} / 04</div>
                        </div>
                        <div class="module-title" style="color:{module['color']}">{module['name']}</div>
                        <div class="module-desc">{module['desc']}</div>
                        <div class="module-divider"></div>
                        <div class="module-label">Yang dipelajari</div>
                        <div class="module-study">{module['lessons']}</div>
                        <div class="module-outcome" style="background:{module['outcome_bg']}">
                          <div class="module-outcome-label">HASIL ANALISIS</div>
                          <div class="module-outcome-text" style="color:{module['color']}">{module['outcome']}</div>
                        </div>
                        <div class="module-foot">{module['foot']}</div>
                        ''', unsafe_allow_html=True)
                        actions = st.columns(len(module["actions"]))
                        for button_col, (idx, name) in zip(actions, module["actions"]):
                            with button_col:
                                st.button(name, key=f"overview_{idx}", icon=":material/arrow_forward:",
                                          use_container_width=True, on_click=select_nav, args=(idx,))


        st.markdown('''<div class="section-head">
  <div class="section-kicker">FROM DATA TO INSIGHT</div>
  <div class="section-title">How Recommendation Systems Work</div>
  <div class="section-desc">Dari data awal hingga rekomendasi yang dapat digunakan.</div>
</div>
<div class="journey-shell">
  <div class="journey-head">
    <div>
      <div class="journey-heading">Dari Data Menjadi Rekomendasi</div>
      <div class="journey-subheading">Lima tahapan inti dalam proses analisis</div>
    </div>
    <div class="journey-pill">5 LANGKAH · 1 ALUR</div>
  </div>
  <div class="journey-grid">
    <div class="journey-step">
      <div class="journey-node"><svg viewBox="0 0 24 24" aria-hidden="true"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg></div>
      <div class="journey-text"><div class="journey-num">STEP 01</div><div class="journey-name">Dataset</div>
      <div class="journey-description">Kumpulkan data transaksi, konten, atau rating.</div></div>
    </div>
    <div class="journey-step">
      <div class="journey-node"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 5h18l-7 8v5l-4 2v-7L3 5Z"/><path d="M8 5v3"/></svg></div>
      <div class="journey-text"><div class="journey-num">STEP 02</div><div class="journey-name">Persiapan Data</div>
      <div class="journey-description">Bersihkan dan susun data agar siap dianalisis.</div></div>
    </div>
    <div class="journey-step">
      <div class="journey-node"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="2.5" fill="none"/><circle cx="15" cy="17" r="2.5" fill="none"/></svg></div>
      <div class="journey-text"><div class="journey-num">STEP 03</div><div class="journey-name">Pemilihan Metode</div>
      <div class="journey-description">Terapkan Apriori, Content-Based, atau Collaborative.</div></div>
    </div>
    <div class="journey-step">
      <div class="journey-node"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20h16M6 17v-6M12 17V5M18 17V9"/></svg></div>
      <div class="journey-text"><div class="journey-num">STEP 04</div><div class="journey-name">Analisis Hasil</div>
      <div class="journey-description">Evaluasi pola, kemiripan, atau skor rekomendasi.</div></div>
    </div>
    <div class="journey-step">
      <div class="journey-node"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 3 2.8 5.7 6.2.9-4.5 4.4 1.1 6.2L12 17.3l-5.6 2.9 1.1-6.2L3 9.6l6.2-.9L12 3Z"/></svg></div>
      <div class="journey-text"><div class="journey-num">STEP 05</div><div class="journey-name">Rekomendasi</div>
      <div class="journey-description">Tampilkan barang atau film yang paling relevan.</div></div>
    </div>
  </div>
  <div class="journey-footer"><span class="journey-footer-dot"></span><span>Alur ini merupakan gambaran umum. Tahapan rinci menyesuaikan metode yang digunakan pada setiap modul.</span></div>
</div>''', unsafe_allow_html=True)

        st.markdown('''<div class="section-head">
          <div class="section-kicker">DATA SOURCES</div>
          <div class="section-title">Informasi Dataset</div>
          <div class="section-desc">Ringkasan data yang digunakan pada masing-masing aktivitas praktik.</div>
        </div>''', unsafe_allow_html=True)
        groceries = groceries_raw()
        movies = movies_raw()
        books = book_interactions()
        dataset_info = pd.DataFrame([
            ["Apriori contoh", "Data transaksi pada materi", 5, "Transaksi sederhana"],
            ["Groceries", "groceries.csv", len(groceries), "Baris transaksi"],
            ["MovieLens", "movie.csv", len(movies), "Film"],
            ["MovieLens rating", "rating.csv", len(ratings_raw()), "Rating tambahan dalam arsip"],
            ["Book-Crossing — data tersaring", "book_knn_interactions.csv.gz", len(books), "Interaksi rating terpilih"],
            ["User-Based dan Hybrid", "Data contoh pada materi", len(EXAMPLE_RATINGS), "Baris rating contoh"],
        ], columns=["Dataset", "Sumber", "Jumlah baris", "Keterangan"])
        compact_dataframe(dataset_info, hide_index=True, use_container_width=True)
        st.caption(
            "File music.csv dan Online Retail.xlsx tersedia dalam paket materi, namun tidak dipakai langsung "
            "oleh tujuh aktivitas yang ditampilkan."
        )

def render_apriori():
    hero('AKTIVITAS 01 · ASSOCIATION RULE', 'Apriori pada transaksi contoh', 'Mencari kombinasi barang yang sering dibeli bersama menggunakan lima transaksi berdasarkan materi.')
    section_heading("PENGATURAN ANALISIS", "Parameter dan Dataset", "Sesuaikan input sesuai kebutuhan sebelum mengeksplorasi hasil.")
    up=st.file_uploader('Opsional: unggah CSV transaksi sendiri (tiap baris = satu transaksi; tanpa header)', type=['csv'], key='up_a1')
    if up:
        tx=transactions_from_frame(pd.read_csv(BytesIO(up.getvalue()),header=None,dtype=str))
    else: tx=EXAMPLE_BASKET
    if not tx:
        st.warning('CSV tidak memiliki transaksi yang dapat dianalisis.');return
    a,b=st.columns(2)
    with a: support=st.slider('Minimum support',0.1,1.0,0.4,0.05,key='sup_a1')
    with b: confidence=st.slider('Minimum confidence',0.1,1.0,0.7,0.05,key='conf_a1')
    itemsets,rules,basket=apriori_tables(tx,support,confidence)
    section_heading("HASIL ANALISIS", "Ringkasan Hasil", "Indikator utama berdasarkan parameter yang dipilih.")
    metrics([('Transaksi',f'{len(tx):,}'),('Jenis barang',str(basket.shape[1])),('Frequent itemsets',str(len(itemsets))),('Association rules',str(len(rules)))])
    t1,t2,t3=st.tabs(['Data transaksi','Frequent itemsets','Aturan asosiasi'])
    with t1:
        compact_dataframe(pd.DataFrame({'Transaksi':[i+1 for i in range(len(tx))], 'Isi keranjang':[', '.join(t) for t in tx]}), hide_index=True,use_container_width=True)
        compact_dataframe(basket.head(15),use_container_width=True)
    with t2:
        st.markdown('Kombinasi barang yang memiliki nilai support ≥ batas pilihan.')
        show_table(pretty_itemsets(itemsets),'apriori_frequent_itemsets.csv')
    with t3:
        display=pretty_rules(rules)
        if display.empty: st.info('Tidak ada aturan untuk parameter yang dipilih. Coba turunkan minimum confidence atau support.')
        else:
            show_table(display,'apriori_rules.csv')
            
            graph = display.head(12).iloc[::-1].copy()
            graph['Aturan'] = graph['Jika membeli'] + ' → ' + graph['Maka membeli']

            fig = px.bar(
                graph,
                x='Lift',
                y='Aturan',
                color='Confidence',
                orientation='h',
                text='Lift',
                hover_data=['Support', 'Confidence'],
                title='Aturan dengan nilai lift terbesar'
            )
            
            fig.update_yaxes(
                categoryorder='array',
                categoryarray=graph['Aturan'].tolist()
            )

            fig.update_traces(
                texttemplate='%{text:.2f}',
                textposition='outside'
            )

            chart_layout(fig, height=420)

    note('<b>Interpretasi:</b> support menunjukkan proporsi transaksi yang memuat kombinasi barang; confidence menunjukkan peluang barang konsekuen muncul saat anteseden muncul; lift &gt; 1 menunjukkan asosiasi positif relatif terhadap peluang dasarnya.')


def render_groceries():
    hero('AKTIVITAS 02 · ASSOCIATION RULE', 'Market Basket Analysis (Groceries)',
         'Eksplorasi transaksi belanja dan aturan Apriori menggunakan dataset Groceries berdasarkan materi.')
    section_heading("PENGATURAN ANALISIS", "Parameter dan Dataset", "Sesuaikan input sesuai kebutuhan sebelum mengeksplorasi hasil.")
    st.info('Mode aman: membuka halaman ini tidak otomatis menjalankan perhitungan. Atur parameter, lalu klik **Jalankan Analisis**.')

    
    with st.container(border=True):
        up = st.file_uploader(
            'Opsional: ganti data dengan CSV transaksi (tanpa header)',
            type=['csv'],
            key='up_gro'
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            support = st.number_input(
                'Minimum support',
                min_value=0.001,
                max_value=0.1,
                value=0.005,
                step=0.001,
                format='%.3f',
                key='sup_g',
                help='0,005 berarti kombinasi muncul pada setidaknya 0,5% transaksi.'
            )

        with c2:
            confidence = st.slider(
                'Minimum confidence',
                min_value=0.05,
                max_value=1.0,
                value=0.8,
                step=0.05,
                key='conf_g'
            )

        with c3:
            max_len = st.selectbox(
                'Panjang kombinasi maksimal',
                [2, 3],
                index=0,
                key='maxlen_g',
                help='Dibatasi pada 2 atau 3 item untuk menjaga penggunaan RAM di Streamlit Cloud.'
            )

        submitted = st.button(
            '▶ Jalankan Analisis',
            type='primary',
            use_container_width=True,
            key='run_groceries'
        )

    if submitted:
        try:
            with st.spinner('Menghitung aturan asosiasi. Proses berjalan dengan batas memori...'):
                res = run_groceries(
                    float(support),
                    float(confidence),
                    int(max_len),
                    up.getvalue() if up is not None else None
                )

            st.session_state['groceries_last_result'] = res
            st.session_state['groceries_last_params'] = (
                float(support),
                float(confidence),
                int(max_len)
            )

            st.success('Analisis berhasil dijalankan.')

        except Exception as e:
            st.error(f'Perhitungan gagal: {e}')
            return

    if 'groceries_last_result' not in st.session_state:
        st.caption(
            'Dataset default: 9.835 transaksi dan 169 jenis barang. '
            'Gunakan parameter bawaan untuk uji pertama.'
        )
        return


    itemsets,rules,frequency,n_orders,n_items = st.session_state['groceries_last_result']
    s_prev,c_prev,l_prev = st.session_state['groceries_last_params']
    st.caption(f'Hasil perhitungan terakhir — support {s_prev:.3f} · confidence {c_prev:.2f} · maksimal {l_prev} barang per kombinasi.')
    section_heading("HASIL ANALISIS", "Ringkasan Hasil", "Indikator utama berdasarkan parameter yang dipilih.")
    metrics([('Transaksi',f'{n_orders:,}'),('Jenis barang',f'{n_items:,}'),
             ('Frequent itemsets',f'{len(itemsets):,}'),('Aturan asosiasi',f'{len(rules):,}')])
    t1,t2,t3 = st.tabs(['Frekuensi barang','Aturan asosiasi','Analisis barang'])
    with t1:
        top=frequency.head(15).rename_axis('Barang').reset_index(name='Frekuensi')
        chart_layout(px.bar(top,x='Frekuensi',y='Barang',orientation='h',title='15 jenis barang paling sering muncul'))
        compact_dataframe(top,hide_index=True,use_container_width=True)
    with t2:
        display=pretty_rules(rules)
        if display.empty:
            st.info('Belum ada aturan yang memenuhi batas support dan confidence. Turunkan parameternya.')
        else:
            show_table(display.head(250),'groceries_rules_top250.csv')
            graph=display.head(300)
            chart_layout(px.scatter(graph,x='Support',y='Confidence',size='Lift',
                                    hover_data=['Jika membeli','Maka membeli'],
                                    title='Pola support–confidence–lift'))
    with t3:
        display=pretty_rules(rules)
        prod=st.selectbox('Pilih barang',list(frequency.index),key='target_g')
        direction=st.radio('Posisi barang', ['Barang yang disarankan (RHS)','Barang yang sudah dipilih (LHS)'],horizontal=True)
        if display.empty:
            st.info('Belum ada aturan yang memenuhi parameter.')
        else:
            col='Maka membeli' if direction.startswith('Barang yang disarankan') else 'Jika membeli'
            filtered=display[display[col].apply(lambda s:prod in [x.strip() for x in s.split(',')])]
            if len(filtered):show_table(filtered.head(30),'aturan_barang_spesifik.csv')
            else:st.info('Tidak ditemukan aturan untuk barang tersebut pada parameter saat ini.')
    note('Dataset yang digunakan adalah CSV Groceries dalam materi. Batas panjang kombinasi 2 atau 3 diterapkan untuk mencegah aplikasi kehabisan memori; karena itu jumlah aturan mungkin berbeda dari pengolahan tanpa batas di R.')


def render_content():
    hero('AKTIVITAS 03 · CONTENT-BASED', 'Content-Based Recommendation', 'Rekomendasi berdasarkan kemiripan deskripsi lima film menggunakan TF-IDF dan cosine similarity.')
    section_heading("PENGATURAN ANALISIS", "Parameter dan Dataset", "Sesuaikan input sesuai kebutuhan sebelum mengeksplorasi hasil.")
    st.caption('Default: lima film contoh yang ditulis langsung dalam notebook Content_Based_Recommendation.ipynb.')
    up=st.file_uploader('Opsional: unggah CSV dengan kolom Judul dan Deskripsi',type=['csv'],key='up_cb')
    df=pd.read_csv(up) if up else EXAMPLE_CONTENT.copy()
    if not {'Judul','Deskripsi'}.issubset(df.columns): st.error('CSV memerlukan kolom Judul dan Deskripsi.');return
    df=df[['Judul','Deskripsi']].dropna().drop_duplicates(subset=['Judul']).reset_index(drop=True)
    if len(df)<2: st.warning('Setidaknya diperlukan 2 judul film.');return
    try:
        matrix=TfidfVectorizer(stop_words='english').fit_transform(df['Deskripsi'].astype(str))
        similarity=cosine_similarity(matrix)
    except ValueError as e:st.error(f'Deskripsi belum bisa diolah: {e}');return
    left,right=st.columns([2,1])
    with left: title=st.selectbox('Pilih film yang disukai',df.Judul)
    with right: n=st.slider('Jumlah rekomendasi',1,min(len(df)-1,15),min(2,len(df)-1))
    idx=df.index[df.Judul==title][0]
    nearest=np.argsort(-similarity[idx],kind='stable')
    nearest=[i for i in nearest if i!=idx][:n]
    result=pd.DataFrame({'Film rekomendasi':df.loc[nearest,'Judul'].to_numpy(),'Kemiripan':similarity[idx,nearest]})
    section_heading("HASIL ANALISIS", "Ringkasan Hasil", "Indikator utama berdasarkan parameter yang dipilih.")
    metrics([('Jumlah film',len(df)),('Istilah TF-IDF',matrix.shape[1]),('Rekomendasi',len(result))])
    t1,t2,t3=st.tabs(['Rekomendasi','Matriks kemiripan','Data deskripsi'])
    with t1:
        show_table(result,'content_based_rekomendasi.csv')
        chart_layout(px.bar(result,x='Kemiripan',y='Film rekomendasi',orientation='h',title=f'Film mirip dengan {title}'))
    with t2:
        if len(df)<=30:
            chart_layout(px.imshow(similarity,x=df.Judul.tolist(),y=df.Judul.tolist(),zmin=0,zmax=1,aspect='auto',title='Cosine similarity antarfilm'),height=390)
        else: st.info('Matriks heatmap dibatasi untuk maksimal 30 film; rekomendasi tetap diproses.')
    with t3: compact_dataframe(df,use_container_width=True,hide_index=True)
    note('TF-IDF mengubah deskripsi menjadi representasi angka, lalu cosine similarity mengukur kedekatan konten. Skor tinggi berarti deskripsi lebih mirip, bukan jaminan film akan disukai pengguna.')


def render_movie():
    hero('AKTIVITAS 04 · CONTENT-BASED', 'Movie Recommendation (MovieLens)', 'Rekomendasi film berdasarkan kemiripan genre dari dataset Movie.zip yang digunakan dalam materi.')
    section_heading("PENGATURAN ANALISIS", "Parameter dan Dataset", "Sesuaikan input sesuai kebutuhan sebelum mengeksplorasi hasil.")
    up=st.file_uploader('Opsional: unggah CSV dengan kolom movieId, title, genres',type=['csv'],key='up_mov')
    if up:
        frame=pd.read_csv(up)
        if not {'title','genres'}.issubset(frame.columns):st.error('CSV harus memiliki minimal kolom title dan genres.');return
        frame=frame[frame.genres.fillna('(no genres listed)')!='(no genres listed)'].dropna(subset=['title','genres']).reset_index(drop=True)
        frame['title_year']=frame.title.astype(str)
        frame['year']=pd.to_numeric(frame.title.str.extract(r'\((\d{4})\)$')[0],errors='coerce')
        frame['title_clean']=frame.title_year.str.replace(r'\s*\(\d{4}\)\s*$', '',regex=True)
        vect=TfidfVectorizer(stop_words='english')
        X=vect.fit_transform(frame.genres.astype(str).str.replace('|',' ',regex=False).str.replace('Sci-Fi','Scifi',regex=False).str.replace('Film-Noir','Noir',regex=False))
    else:frame,X=prep_movies()
    if len(frame)<2:st.warning('Data film tidak cukup.');return
    title=st.selectbox('Pilih judul film (bisa diketik untuk mencari)',frame.title_year.tolist())
    n=st.slider('Jumlah rekomendasi',3,20,10)
    ix=int(frame.index[frame.title_year==title][0]); scores=linear_kernel(X[ix],X).ravel();scores[ix]=-1
    top=np.argsort(-scores,kind='stable')[:n]
    result=frame.loc[top,['title_year','genres','year']].copy()
    result['Kesamaan genre']=scores[top]
    result=result.rename(columns={'title_year':'Film rekomendasi','genres':'Genre','year':'Tahun'})
    section_heading("HASIL ANALISIS", "Ringkasan Hasil", "Indikator utama berdasarkan parameter yang dipilih.")
    metrics([('Film valid',f'{len(frame):,}'),('Fitur genre TF-IDF',str(X.shape[1])),('Rekomendasi',len(result))])
    tab1,tab2,tab3=st.tabs(['Rekomendasi film','Analisis genre','Dataset'])
    with tab1:
        show_table(result,'movie_recommendations.csv')
        chart_layout(px.bar(result.head(12),x='Kesamaan genre',y='Film rekomendasi',orientation='h',title=f'Film paling mirip dengan {title}'),height=400)
    with tab2:
        from collections import Counter
        counts=Counter(g.strip() for val in frame.genres.dropna() for g in str(val).split('|') if g.strip())
        top_genres=pd.DataFrame(counts.most_common(18),columns=['Genre','Jumlah film'])
        chart_layout(px.bar(top_genres,x='Jumlah film',y='Genre',orientation='h',title='18 genre paling umum'))
        compact_dataframe(top_genres,hide_index=True,use_container_width=True)
    with tab3:
        st.markdown('**Cari film dalam dataset MovieLens**')
        st.caption('Ketik sebagian judul untuk mencari di seluruh katalog film. Data ini memuat judul, genre, dan tahun; bukan sinopsis cerita.')
        search_title = st.text_input(
            'Cari judul film',
            placeholder='Contoh: Inception, Toy Story, Titanic',
            key='movie_dataset_search',
        ).strip()
        # Pencarian berlaku untuk seluruh film valid, tidak hanya 200 baris awal.
        dataset_view = frame[['title_year', 'genres', 'year']]
        if search_title:
            dataset_view = dataset_view[
                dataset_view['title_year'].str.contains(search_title, case=False, regex=False, na=False)
            ]
        else:
            st.caption(f'Film acuan saat ini: **{title}** | Genre: **{frame.loc[ix, "genres"]}**')

        st.caption(f'**{len(dataset_view):,} film** cocok dengan pencarian.' if search_title
                   else f'**{len(dataset_view):,} film** tersedia dalam dataset.')
        if dataset_view.empty:
            st.info('Judul tidak ditemukan. Coba gunakan sebagian kata dari judul film.')
        else:
            compact_dataframe(
                dataset_view.head(50).rename(columns={
                    'title_year': 'Judul film', 'genres': 'Genre', 'year': 'Tahun',
                }),
                hide_index=True, use_container_width=True,
            )
            if len(dataset_view) > 50:
                st.caption('Menampilkan 50 hasil pertama. Ketik judul lebih spesifik untuk mempersempit hasil.')
        st.caption('Rating.csv juga tersedia dari Movie.zip, tetapi notebook Content_Based_Filtering_ML menggunakan fitur genre dari movie.csv untuk metode ini.')
    note('Untuk menghemat memori, dashboard menghitung kemiripan film yang dipilih terhadap semua film, bukan membangun matriks kemiripan penuh 9.000 × 9.000. Rumus cosine similarity tetap setara.')


def render_user_cf():
    hero('AKTIVITAS 05 · COLLABORATIVE FILTERING', 'User-Based Collaborative Filtering', 'Mencari tetangga pengguna paling mirip dari pola rating film yang dicontohkan berdasarkan materi.')
    section_heading("PENGATURAN ANALISIS", "Parameter dan Dataset", "Sesuaikan input sesuai kebutuhan sebelum mengeksplorasi hasil.")

    # Data bawaan tetap dipertahankan. Pilihan demo disediakan supaya hasil
    # rekomendasi bisa diuji tanpa mengubah dataset asli dari materi.
    demo = EXAMPLE_RATINGS.copy()
    demo = pd.concat([demo, pd.DataFrame([
        {'Pengguna': 'Budi', 'Film': 'Interstellar', 'Rating': 4}
    ])], ignore_index=True)
    demo_csv = demo.to_csv(index=False).encode('utf-8-sig')
    st.download_button('Unduh CSV demo (ada rekomendasi)', data=demo_csv,
                       file_name='contoh_user_based_rekomendasi.csv', mime='text/csv',
                       icon=':material/download:', key='download_demo_user_cf')
    up = st.file_uploader('Opsional: unggah CSV dengan kolom Pengguna, Film, Rating',
                          type=['csv'], key='up_ub')
    use_demo = st.checkbox('Coba data demo yang menghasilkan rekomendasi (tanpa upload)',
                            value=False, key='use_demo_user_cf', disabled=up is not None)

    if up is not None:
        try:
            df = pd.read_csv(BytesIO(up.getvalue()), sep=None, engine='python', encoding='utf-8-sig')
        except (ValueError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            st.error(f'CSV tidak dapat dibaca: {exc}')
            return
        source = f'File CSV: {up.name}'
    elif use_demo:
        df = demo.copy()
        source = 'Data demo rekomendasi (tambahan rating Budi untuk Interstellar)'
    else:
        df = EXAMPLE_RATINGS.copy()
        source = 'Data bawaan dari materi (tanpa film tambahan untuk Budi)'

    df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '', regex=False)
    if not {'Pengguna', 'Film', 'Rating'}.issubset(df.columns):
        st.error('CSV harus memiliki kolom Pengguna, Film, Rating. '
                 f'Kolom yang terbaca: {", ".join(map(str, df.columns))}.')
        return
    df = df[['Pengguna', 'Film', 'Rating']].dropna().copy()
    df['Pengguna'] = df['Pengguna'].astype(str).str.strip()
    df['Film'] = df['Film'].astype(str).str.strip()
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df = df.dropna(subset=['Rating'])
    df = df[(df['Pengguna'] != '') & (df['Film'] != '')]
    if df.empty or df.Pengguna.nunique() < 2:
        st.warning('Dibutuhkan minimal dua pengguna dengan data rating yang valid.')
        return
    # Tampilkan status data aktif agar jelas bahwa upload menggantikan data bawaan.
    st.success(f'Dataset aktif: **{source}** · {len(df)} baris rating · '
               f'{df.Pengguna.nunique()} pengguna · {df.Film.nunique()} film.')
    with st.expander('Periksa data rating yang sedang digunakan', expanded=False):
        compact_dataframe(df.head(30), hide_index=True, use_container_width=True)

    matrix, sim = ratings_matrix(df)
    left, right = st.columns(2)
    with left:
        user = st.selectbox('Pilih pengguna', matrix.index, key='selected_user_cf')
    with right:
        threshold = st.slider('Minimal rating dari pengguna serupa', 1, 5, 4, key='min_rating_user_cf')
    neighbours = sim[user].drop(user).sort_values(ascending=False)
    nearest = neighbours.index[0]
    unseen = (matrix.loc[user] == 0) & (matrix.loc[nearest] >= threshold)
    recommendations = pd.DataFrame({
        'Film': matrix.columns[unseen],
        'Rating pengguna serupa': matrix.loc[nearest, unseen].values,
    })
    section_heading("HASIL ANALISIS", "Ringkasan Hasil", "Indikator utama berdasarkan parameter yang dipilih.")
    metrics([('Pengguna', df.Pengguna.nunique()), ('Film', df.Film.nunique()),
             ('Tetangga terdekat', nearest), ('Kemiripan', f'{float(neighbours.iloc[0]):.3f}')])
    t1, t2, t3 = st.tabs(['Hasil rekomendasi', 'Kemiripan pengguna', 'Matriks rating'])
    with t1:
        st.write(f'Pengguna **{user}** paling mirip dengan **{nearest}**.')
        if len(recommendations):
            show_table(recommendations, 'user_based_rekomendasi.csv')
        else:
            st.info('Tidak ada film baru dengan rating tetangga yang memenuhi batas ini. '
                    'Untuk melihat hasil demo, centang opsi data demo di atas, '
                    'pilih Andi, dan gunakan minimal rating 4.')
    with t2:
        chart_layout(px.imshow(sim, x=sim.columns, y=sim.index, zmin=0, zmax=1,
                               title='Cosine similarity antar pengguna'))
        compact_dataframe(neighbours.rename('Skor kesamaan').reset_index().rename(
            columns={'index': 'Pengguna lain'}), hide_index=True, use_container_width=True)
    with t3:
        compact_dataframe(matrix, use_container_width=True)
    note('Angka 0 pada matriks berarti <b>belum ada rating</b> (missing), bukan rating negatif. '
         'Penentuan rekomendasi mengikuti notebook: film yang belum dinilai pengguna target '
         'tetapi diberi rating tinggi oleh tetangga terdekat.')


def render_item_cf():
    hero('AKTIVITAS 06 · COLLABORATIVE FILTERING', 'Item-Based kNN (Book-Crossing)', 'Menemukan judul buku yang serupa dari pola rating pengguna menggunakan k-Nearest Neighbors dengan jarak cosine.')
    section_heading("PENGATURAN ANALISIS", "Parameter dan Dataset", "Sesuaikan input sesuai kebutuhan sebelum mengeksplorasi hasil.")
    st.caption('Dataset siap pakai telah diproses dari tiga CSV dalam BX-Users.zip menggunakan filter yang tercantum di materi.')
    up=st.file_uploader('Opsional: unggah CSV siap olah dengan kolom userID, bookTitle, bookRating',type=['csv'],key='up_book')
    if up:
        df=pd.read_csv(up)
        if not {'userID','bookTitle','bookRating'}.issubset(df.columns): st.error('CSV harus memiliki kolom userID, bookTitle, bookRating.');return
        df=df[['userID','bookTitle','bookRating']].dropna()
        df['bookRating']=pd.to_numeric(df.bookRating,errors='coerce')
        df=df.dropna(subset=['bookRating']).drop_duplicates(['userID','bookTitle'])
        pivot=df.pivot_table(index='bookTitle',columns='userID',values='bookRating',aggfunc='mean',fill_value=0)
        sparse=csr_matrix(pivot.to_numpy(dtype=np.float32));model=NearestNeighbors(metric='cosine',algorithm='brute').fit(sparse)
    else:
        df=book_interactions()
        pivot,sparse,model=prep_books()
    if len(pivot)<2:st.warning('Jumlah buku tidak cukup.');return
    name=st.selectbox('Pilih buku yang dijadikan acuan',pivot.index.tolist(),index=min(25,len(pivot)-1))
    n=st.slider('Jumlah rekomendasi',1,min(15,len(pivot)-1),min(5,len(pivot)-1))
    index=int(pivot.index.get_loc(name))
    dists,indices=model.kneighbors(sparse[index],n_neighbors=min(n+1,len(pivot)))
    rec=[]
    for j,d in zip(indices[0],dists[0]):
        if j!=index: rec.append({'Buku rekomendasi':pivot.index[j],'Skor Kemiripan':round(float(1-d),4)})
    result=pd.DataFrame(rec).head(n)
    section_heading("HASIL ANALISIS", "Ringkasan Hasil", "Indikator utama berdasarkan parameter yang dipilih.")
    metrics([('Interaksi setelah filter',f'{len(df):,}'),('Buku',f'{len(pivot):,}'),('Pengguna',f'{len(pivot.columns):,}'),('Rekomendasi',len(result))])
    t1,t2,t3=st.tabs(['Rekomendasi buku','Pola rating','Proses filter data'])
    with t1:
        if len(result):
            show_table(result,'item_based_buku.csv')
            chart_layout(px.bar(result,x='Skor Kemiripan',y='Buku rekomendasi',orientation='h',title=f'Buku yang mirip dengan {name[:65]}'),height=390)
    with t2:
        counts=df.bookRating.value_counts().sort_index().rename_axis('Rating').reset_index(name='Jumlah')
        chart_layout(px.bar(counts,x='Rating',y='Jumlah',title='Distribusi rating buku'))
        st.caption('Rating 0 berasal dari data Book-Crossing dan merepresentasikan rating implisit / tanpa penilaian numerik eksplisit.')
    with t3:
        st.markdown('''**Mengikuti tahapan materi:**

1. Memilih pengguna dengan setidaknya 200 interaksi rating pada data asli.
2. Mempertahankan nilai rating yang muncul sedikitnya 100 kali (sesuai script).
3. Menggabungkan data rating dan judul buku; memilih judul dengan setidaknya 50 interaksi rating.
4. Menyaring pengguna dengan lokasi mengandung `usa` atau `canada`.
5. Menghapus pasangan pengguna–judul yang duplikat, lalu membentuk matriks buku × pengguna untuk kNN cosine.

**Catatan:** agar repo ringan, dashboard menyertakan data hasil penyaringan dari arsip sumber, bukan keseluruhan arsip mentah. Jumlah buku dapat berbeda bila CSV dibaca dengan kebijakan baris rusak yang berbeda.''')
    note('Semakin tinggi <b>Skor Kemiripan</b> (mendekati 1), semakin mirip pola rating kedua buku. Skor ini dihitung dari cosine similarity berdasarkan rating pembaca, bukan kesamaan isi cerita buku.')


def render_hybrid():
    hero('AKTIVITAS 07 · HYBRID', 'Hybrid Recommendation System', 'Mengombinasikan content-based filtering (kemiripan genre) dan collaborative filtering (pola rating pengguna).')
    section_heading("PENGATURAN ANALISIS", "Parameter dan Dataset", "Sesuaikan input sesuai kebutuhan sebelum mengeksplorasi hasil.")
    movies=EXAMPLE_HYBRID.copy()
    ratings=EXAMPLE_RATINGS.rename(columns={'Pengguna':'User','Film':'Movie'}).copy()
    a,b=st.columns(2)
    with a: target=st.selectbox('Pengguna target',sorted(ratings.User.unique()),index=1)
    with b: cb_weight=st.slider('Bobot Content-Based',0.0,1.0,0.5,0.05)
    cf_weight=1-cb_weight
    tfidf=TfidfVectorizer()
    X=tfidf.fit_transform(movies.Genre)
    item_sim=pd.DataFrame(cosine_similarity(X),index=movies.Movie,columns=movies.Movie)
    matrix=ratings.pivot_table(index='User',columns='Movie',values='Rating').fillna(0)
    user_sim=pd.DataFrame(cosine_similarity(matrix),index=matrix.index,columns=matrix.index)
    liked=ratings.loc[(ratings.User==target)&(ratings.Rating>=4),'Movie'].tolist()
    rated=ratings.loc[ratings.User==target,'Movie'].tolist()
    candidate=[movie for movie in movies.Movie if movie not in rated]
    scores=[]
    for movie in candidate:
        cb=float(np.mean([item_sim.loc[movie,x] for x in liked])) if liked else 0.0
        cf=0.0
        for other,sim in user_sim[target].drop(target).sort_values(ascending=False).items():
            if movie in matrix.columns and matrix.loc[other,movie]>0:
                cf=float(matrix.loc[other,movie]*sim)
                break
        scores.append({'Film':movie,'CB Score':cb,'CF Score':cf})
    score_df=pd.DataFrame(scores)
    if not score_df.empty:
        score_df[['CB Normalisasi','CF Normalisasi']]=MinMaxScaler().fit_transform(score_df[['CB Score','CF Score']])
        score_df['Hybrid Score']=cb_weight*score_df['CB Normalisasi']+cf_weight*score_df['CF Normalisasi']
        score_df=score_df.sort_values('Hybrid Score',ascending=False).reset_index(drop=True)
    section_heading("HASIL ANALISIS", "Ringkasan Hasil", "Indikator utama berdasarkan parameter yang dipilih.")
    metrics([('Pengguna',target),('Film kandidat',len(candidate)),('Bobot CB',f'{cb_weight:.0%}'),('Bobot CF',f'{cf_weight:.0%}')])
    t1,t2,t3=st.tabs(['Peringkat rekomendasi','Kontribusi metode','Data dan rumus'])
    with t1:
        if score_df.empty: st.info('Pengguna sudah memberi rating pada semua film; tidak ada kandidat baru.')
        else:
            show_table(score_df,'hybrid_rekomendasi.csv')
            chart_layout(px.bar(score_df,x='Hybrid Score',y='Film',orientation='h',title=f'Rekomendasi untuk {target}'))
    with t2:
        if not score_df.empty:
            plot=score_df[['Film','CB Normalisasi','CF Normalisasi']].melt(id_vars='Film',var_name='Komponen',value_name='Skor normalisasi')
            chart_layout(px.bar(plot,x='Film',y='Skor normalisasi',color='Komponen',barmode='group',title='Skor tiap metode sebelum diberi bobot'))
    with t3:
        st.markdown('''**Perhitungan (sesuai notebook):**

- **CB Score** = rata-rata kesamaan genre film kandidat terhadap film yang disukai pengguna (rating ≥ 4).
- **CF Score** = rating tetangga terdekat yang pernah menilai film kandidat × kemiripan pengguna.
- **Normalisasi** = MinMaxScaler, dilakukan terpisah untuk CB dan CF pada film kandidat.
- **Hybrid Score** = `(bobot CB × CB Normalisasi) + (bobot CF × CF Normalisasi)`.

Skor merupakan peringkat internal contoh ini, **bukan probabilitas** seorang pengguna akan menyukai film.''')
        compact_dataframe(movies,hide_index=True,use_container_width=True)
        compact_dataframe(ratings,hide_index=True,use_container_width=True)
    note('Data dan formula dipertahankan dari notebook Hybrid pada materi. Bobot dapat diganti untuk melihat pengaruh pendekatan berbasis konten dan pengguna terhadap urutan rekomendasi.')


if page==NAV[0]: render_overview()
elif page==NAV[1]: render_apriori()
elif page==NAV[2]: render_groceries()
elif page==NAV[3]: render_content()
elif page==NAV[4]: render_movie()
elif page==NAV[5]: render_user_cf()
elif page==NAV[6]: render_item_cf()
elif page==NAV[7]: render_hybrid()
