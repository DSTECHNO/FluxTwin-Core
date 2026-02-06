#    ______  _            _______          _
#   |  ____|| |          |__   __|        (_)
#   | |__   | | __ ___  __  | | __      __ _ _ __
#   |  __|  | |/ _` \ \/ /  | | \ \ /\ / /| | '_ \
#   | |     | | (_| |>  <   | |  \ V  V / | | | | |
#   |_|     |_|\__,_/_/\_\  |_|   \_/\_/  |_|_| |_|
#
# ================================================================
# 📄 File Name     : app.py
# 🧩 Component     : FluxTwin Core – Streamlit Orchestrator
# 🌍 Platform      : FluxTwin – CFD-Based Digital Twin Framework
# 🌐 Deployment    : https://fluxtwin-core.dstechs.net/
# ⚖️ License       : AGPLv3 (Open Source)
# 👨‍💻 Developed By  : D&S Tech
# 📅 Date Created  : 06.02.2026
# 🔄 Last Update   : 06.02.2026
#
# 📌 Overview:
#   This script is the main entry point of the FluxTwin web platform.
#   It acts as a lightweight orchestration layer that connects multiple
#   pilot data centres to a unified CFD visualization interface.
#
#   Responsibilities:
#     • Initialize the Streamlit application (layout, branding, UX flow)
#     • Provide a landing interface for pilot data centre selection
#     • Route user interaction to site-specific CFD viewers
#     • Enable scalable integration of new digital-twin deployments
#
#   FluxTwin demonstrates how high-fidelity CFD simulations can be
#   transformed into an interactive, browser-based decision-support
#   environment for data centre thermal and airflow analysis.
#
# 🎯 Target Audience:
#   • Data centre operators and facility managers
#   • Thermal / CFD engineers
#   • Digital twin and energy optimization stakeholders
#
# ✉️ Contact:
#   Data Centre Digital Twin Solutions
#   D&S Tech — datacenter@dstechs.net
# ================================================================

import streamlit as st
import runpy
from pathlib import Path
import base64

# ---------- Helper function ----------
def file_to_data_uri(filepath, mime_type):
    """Convert local file to base64 data URI"""
    if not Path(filepath).exists():
        return None
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime_type};base64,{data}"

# ---------- Page config ----------
st.set_page_config(
    page_title="FluxTwin",
    page_icon="images/FluxTwin_icon.png",
    layout="wide"
)

# ---------- CSS: layout fixes ----------
st.markdown(
    """
    <style>
      /* Main page top padding */
      .block-container { padding-top: 2.5rem; }
      
      /* Sidebar spacing */
      [data-testid="stSidebar"] > div:first-child { padding-top: 0rem !important; }
      [data-testid="stSidebar"] .block-container { padding-top: 0rem !important; }
      [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.2rem !important; }
      
      /* Our custom "Select Pilot Site" title */
      .dc-top-title{
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 12px;
      }
      .spacer-sm { height: 10px; }
      .spacer-md { height: 20px; }
      .spacer-lg { height: 80px; }
      
      /* Alert spacing */
      [data-testid="stSidebar"] [data-testid="stAlert"] {
        margin-top: 0px !important;
        margin-bottom: 0px !important;
      }
      
      /* Button spacing */
      [data-testid="stSidebar"] button { margin-top: 0px !important; }
      
      /* hr spacing */
      [data-testid="stSidebar"] hr { margin-top: 8px !important; margin-bottom: 8px !important; }
      
      /* Text colors that work in both light and dark mode */
      .main-title {
        color: var(--text-color) !important;
      }
      .sub-title {
        color: var(--text-color) !important;
        opacity: 0.7;
      }
      .feature-text {
        color: var(--text-color) !important;
      }
      
      /* Feature image container - ONLY large for 2d_slice_x_axis and KPI_table */
      .feature-img-container-large {
        text-align: center;
        padding: 10px;
        margin-top: 10px;
      }
      .feature-img-container-large img {
        width: 100%;
        max-width: 450px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      }
      
      /* Keep TwinFlux_Code small */
      .feature-img-container-small {
        text-align: center;
        padding: 10px;
        margin-top: 10px;
      }
      .feature-img-container-small img {
        width: 100%;
        max-width: 280px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      }
      
      /* Placeholder for missing images */
      .feature-img-placeholder {
        text-align: center;
        padding: 15px;
        margin-top: 15px;
        background: #f0f0f0;
        border-radius: 8px;
        height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #999;
        font-size: 14px;
      }
      
      /* HEATWISE link styling */
      .heatwise-link {
        color: inherit;
        text-decoration: none;
        border-bottom: 2px solid currentColor;
        transition: opacity 0.2s;
      }
      .heatwise-link:hover {
        opacity: 0.7;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Session defaults ----------
if "dc_locked" not in st.session_state:
    st.session_state.dc_locked = False
if "selected_dc" not in st.session_state:
    st.session_state.selected_dc = "AAU"

DC_LIST = {
    "AAU": "🇩🇰 AAU (Aalborg Univ. - Denmark)",
    "EMPA": "🇨🇭 EMPA (Material Science Lab - Switzerland)",
    "PSNC": "🇵🇱 PSNC (Supercomputing Center - Poland)",
    "TOFAS": "🇹🇷 TOFAS (Automotive Factory - Turkiye)"
}

BASE_DIR = Path(__file__).parent

# ---------- Sidebar ----------
with st.sidebar:
    # FluxTwin logo at top
    logo_src = file_to_data_uri(BASE_DIR / "images/FluxTwin_logo.png", "image/png")
    if logo_src:
        st.markdown(
            f"""
            <div style='text-align: center; padding: 0px 0px 20px 0px;'>
                <img src='{logo_src}' alt='FluxTwin' style='width: 100%; max-width: 260px;'>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown("### FluxTwin")
    
    st.markdown('<div class="dc-top-title">Select Pilot Site</div>', unsafe_allow_html=True)
    
    if st.session_state.dc_locked:
        st.info(f"**{DC_LIST[st.session_state.selected_dc]}**")
        st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
        if st.button("Change Pilot Site", use_container_width=True):
            st.session_state.dc_locked = False
            st.rerun()
    else:
        dc_keys = list(DC_LIST.keys())
        dc_labels = list(DC_LIST.values())
        idx = dc_keys.index(st.session_state.selected_dc)
        
        selected_label = st.selectbox(" ", dc_labels, index=idx, label_visibility="collapsed")
        selected_key = dc_keys[dc_labels.index(selected_label)]
        
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
        if st.button("Continue", type="primary", use_container_width=True):
            st.session_state.selected_dc = selected_key
            st.session_state.dc_locked = True
            st.rerun()
        
        # Spacer between button and footer
        st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    
    # Footer section
    if not st.session_state.dc_locked:
        st.sidebar.markdown(
            """
            <p style='font-size: 13px; color:#6b7280; font-weight:600; margin-bottom:6px;'>
                Developed by D&STECH © 2026
            </p>
            <p style='font-size: 12px; color:#6b7280; line-height:1.5; margin-bottom: 15px;'>
                FluxTwin(c) has been developed as part of EU-funded HEATWISE Project. The Heatwise Project has received funding from the European Union's Horizon Europe research and innovation programme under Grant Agreement No 101138491 and the Swiss Secretariat for Education, Research, and Innovation (SERI) under contract No 23.00606.
            </p>
            """,
            unsafe_allow_html=True
        )
        
        logo_col1, logo_col2 = st.sidebar.columns(2)
        heatwise_src = file_to_data_uri(BASE_DIR / "images/heatwise_logo.svg", "image/svg+xml")
        dstech_src   = file_to_data_uri(BASE_DIR / "images/dstech_logo.png", "image/png")
        
        with logo_col1:
            if heatwise_src:
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 10px;'>
                        <a href='https://heatwise.eu' target='_blank'>
                            <img src='{heatwise_src}' alt='HEATWISE' style='width: 100%; max-width: 120px;'>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        with logo_col2:
            if dstech_src:
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 10px;'>
                        <a href='https://dstechs.net/' target='_blank'>
                            <img src='{dstech_src}' alt='D&S Tech' style='width: 100%; max-width: 120px;'>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        st.sidebar.markdown(
            """
            <p style='font-size: 11px; color:#6b7280; text-align: center; margin-top: 10px; margin-bottom: 20px;'>
                ⚖️ AGPLv3 Licensed - Open Source
            </p>
            """,
            unsafe_allow_html=True
        )

# ---------- Main area ----------
if not st.session_state.dc_locked:
    # Main landing page
    st.markdown(
        """
        <div style="text-align:center; padding-top: 20px;">
          <h1 style="font-size: 48px; font-weight: 800; line-height: 1.2; margin-bottom: 20px;">
            FluxTwin Core: Open-Source Thermal Intelligence
          </h1>
          <p style="font-size: 32px; font-weight: 700; margin-bottom: 20px;">
            ⚡ Zero Installation, Instant Insight.
          </p>
          <h2 style="font-size: 18px; opacity: 0.7; margin-bottom: 30px;">
            Validated CFD technology born from the EU-funded <a href="https://heatwise.eu/" target="_blank" class="heatwise-link">HEATWISE</a> project.
          </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    # Convert images to data URIs (with fallback)
    slice_x_src = file_to_data_uri(BASE_DIR / "images/2d_slice_x_axis.png", "image/png")
    kpi_src = file_to_data_uri(BASE_DIR / "images/KPI_table.png", "image/png")
    code_src = file_to_data_uri(BASE_DIR / "images/TwinFlux_Code.png", "image/png")
    
    with col1:
        # 2d_slice_x_axis
        img_html = f'<img src="{slice_x_src}" alt="Airflow Visualization">' if slice_x_src else '<div class="feature-img-placeholder">Image placeholder</div>'
        st.markdown(
            f"""
            <div style="text-align:center; padding: 10px;">
              <div style="font-size: 28px; margin-bottom: 8px;">👁️</div>
              <p style="font-size: 16px; font-weight: 600; margin-bottom: 5px;">Visualize invisible airflow and thermal patterns.</p>
              <div class="feature-img-container-large">
                {img_html}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        # KPI_table
        img_html = f'<img src="{kpi_src}" alt="KPI Table">' if kpi_src else '<div class="feature-img-placeholder">Image placeholder</div>'
        st.markdown(
            f"""
            <div style="text-align:center; padding: 10px;">
              <div style="font-size: 28px; margin-bottom: 8px;">📊</div>
              <p style="font-size: 16px; font-weight: 600; margin-bottom: 5px;">Calculate cooling KPIs instantly.</p>
              <div class="feature-img-container-large">
                {img_html}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        # TwinFlux_Code
        img_html = f'<img src="{code_src}" alt="Open Source Code">' if code_src else '<div class="feature-img-placeholder">Image placeholder</div>'
        st.markdown(
            f"""
            <div style="text-align:center; padding: 10px;">
              <div style="font-size: 28px; margin-bottom: 8px;">🔓</div>
              <p style="font-size: 16px; font-weight: 600; margin-bottom: 5px;">Powered by Open Source (AGPLv3).</p>
              <div class="feature-img-container-small">
                {img_html}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Footer below images - left aligned
    st.markdown(
        """
        <div style="text-align:left; padding: 60px 20px 20px 20px;">
          <p style="font-size: 16px; font-weight: 600;">Developed by D&S Tech © 2026</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.stop()

# ---------- Run selected viewer ----------
viewer_path = f"{st.session_state.selected_dc}/viewer.py"
runpy.run_path(viewer_path, run_name="__main__")
