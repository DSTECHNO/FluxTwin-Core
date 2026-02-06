#    ______  _            _______          _
#   |  ____|| |          |__   __|        (_)
#   | |__   | | __ ___  __  | | __      __ _ _ __
#   |  __|  | |/ _` \ \/ /  | | \ \ /\ / /| | '_ \
#   | |     | | (_| |>  <   | |  \ V  V / | | | | |
#   |_|     |_|\__,_/_/\_\  |_|   \_/\_/  |_|_| |_|
#
# =================================================================
# 📄 File Name     : viewer.py
# 🧩 Component     : FluxTwin Site Viewer – CFD Post-Processing UI
# 🌍 Platform      : FluxTwin – CFD-Based Digital Twin Framework
# 🌐 Deployment    : https://fluxtwin-core.dstechs.net/
# ⚖️ License       : AGPLv3 (Open Source)
# 👨‍💻 Developed By  : D&S Tech
# 📅 Date Created  : 06.02.2026
# 🔄 Last Update   : 06.02.2026
#
# 🏷️ Pilot Site:
#   AAU / EMPA / PSNC / TOFAS
#
# 📌 Overview:
#   This module provides an interactive post-processing and visualization
#   interface for CFD simulations of data centre environments.
#
#   It bridges raw CFD output and engineering insight by enabling
#   intuitive exploration of three-dimensional thermal and airflow
#   fields directly in a web-based environment.
#
# 🔬 Technical Capabilities:
#   • Secure access via Streamlit secrets (authentication layer)
#   • Efficient handling of large CFD datasets stored in NPZ format
#   • Support for volumetric scalar and vector fields
#   • Optional geometric context via VTK-based surface visualization
#   • Interactive 3D scatter plots with adaptive downsampling
#   • Planar slicing with thickness control and cached interpolation
#   • On-the-fly statistical evaluation of CFD fields
#
# 📊 Engineering Use Case:
#   Enables hotspot detection, airflow quality assessment,
#   thermal compliance checks and scenario-based comparison
#   within a CFD-driven digital twin workflow.
#
# ✉️ Contact:
#   Data Centre Digital Twin Solutions
#   D&S Tech — datacenter@dstechs.net
# =================================================================
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pyvista as pv
from scipy.interpolate import griddata
import os
import requests
from pathlib import Path
import base64

BASE_DIR = Path(__file__).resolve().parent

def file_to_data_uri(path: Path, mime: str) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# -------------------------------------------------
# 1. PASSWORD AUTHENTICATION
# -------------------------------------------------
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Please enter the password:",
        type="password",
        on_change=password_entered,
        key="password"
    )

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect. Please try again.")

    return False


if not check_password():
    st.stop()


# -------------------------------------------------
# FILE DOWNLOAD (CACHE)
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def ensure_file(url: str, local_path: str) -> str:
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()

    tmp = local_path + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    os.replace(tmp, local_path)
    return local_path


# -------------------------------------------------
# NPZ + VTK LOAD
# -------------------------------------------------
def load_npz_case(npz_filename: str, vtk_filename: str):
    data = np.load(npz_filename)
    mesh = pv.read(vtk_filename)

    keys = list(data.files)

    # Prefer cell_ fields if available
    if "cell_T" in keys:
        T = data["cell_T"]
    else:
        T = data["T"] if "T" in keys else None

    if "cell_U" in keys:
        U = data["cell_U"]
    else:
        U = data["U"] if "U" in keys else None

    return mesh, T, U


# -------------------------------------------------
# OUTER GEOMETRY (FROM VTK) LOAD
# -------------------------------------------------
@st.cache_data
def load_outer_geometry(vtk_path: str):
    mesh = pv.read(vtk_path)
    surface = mesh.extract_surface()
    surface = surface.triangulate()

    pts = surface.points
    faces = surface.faces.reshape(-1, 4)
    triangles = faces[:, 1:]

    xg = pts[:, 0]
    yg = pts[:, 1]
    zg = pts[:, 2]
    ig = triangles[:, 0]
    jg = triangles[:, 1]
    kg = triangles[:, 2]

    return xg, yg, zg, ig, jg, kg


# -------------------------------------------------
# INTERPOLATION CACHING
# -------------------------------------------------
@st.cache_data
def interpolate_slice(axis1_s, axis2_s, f_s, grid_resolution):
    axis1_min, axis1_max = axis1_s.min(), axis1_s.max()
    axis2_min, axis2_max = axis2_s.min(), axis2_s.max()

    grid_axis1 = np.linspace(axis1_min, axis1_max, grid_resolution)
    grid_axis2 = np.linspace(axis2_min, axis2_max, grid_resolution)
    grid_axis1_mesh, grid_axis2_mesh = np.meshgrid(grid_axis1, grid_axis2)

    grid_field = griddata(
        (axis1_s, axis2_s),
        f_s,
        (grid_axis1_mesh, grid_axis2_mesh),
        method="linear",
        fill_value=np.nan
    )

    nan_mask = np.isnan(grid_field)
    if nan_mask.any():
        grid_field_nearest = griddata(
            (axis1_s, axis2_s),
            f_s,
            (grid_axis1_mesh, grid_axis2_mesh),
            method="nearest"
        )
        grid_field[nan_mask] = grid_field_nearest[nan_mask]

    return grid_axis1, grid_axis2, grid_field


# -------------------------------------------------
# GET COORDS + FIELD (NO UI HERE!)
# -------------------------------------------------
def get_coords_and_field(mesh, T_field, U_field, field_choice: str):
    if field_choice == "Temperature":
        if T_field is None:
            raise ValueError("NPZ does not contain temperature field (T/cell_T).")
        field = T_field - 273.15
        color_label = "T [°C]"
    else:
        if U_field is None:
            raise ValueError("NPZ does not contain velocity field (U/cell_U).")
        field = np.linalg.norm(U_field, axis=1)
        color_label = "|U| [m/s]"

    field = np.asarray(field)
    if field.ndim == 2 and field.shape[1] == 1:
        field = field[:, 0]

    if len(field) == mesh.n_cells:
        pts = mesh.cell_centers().points
    elif len(field) == mesh.n_points:
        pts = mesh.points
    else:
        raise ValueError(
            f"Field length ({len(field)}) != n_cells ({mesh.n_cells}) and != n_points ({mesh.n_points})."
        )

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return x, y, z, field, color_label


# -------------------------------------------------
# DENSITY-AWARE DOWNSAMPLING
# -------------------------------------------------
@st.cache_data
def density_aware_downsample(x, y, z, field, max_points: int, n_side: int = 40):
    N = x.size
    if N <= max_points:
        return x, y, z, field

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    zmin, zmax = float(z.min()), float(z.max())

    ex = xmax - xmin if xmax > xmin else 1e-9
    ey = ymax - ymin if ymax > ymin else 1e-9
    ez = zmax - zmin if zmax > zmin else 1e-9

    ix = ((x - xmin) / ex * n_side).astype(int)
    iy = ((y - ymin) / ey * n_side).astype(int)
    iz = ((z - zmin) / ez * n_side).astype(int)

    ix = np.clip(ix, 0, n_side - 1)
    iy = np.clip(iy, 0, n_side - 1)
    iz = np.clip(iz, 0, n_side - 1)

    key = ix + n_side * (iy + n_side * iz)

    counts = np.bincount(key)
    density = counts[key]

    weights = 1.0 / density
    weights_sum = weights.sum()
    if weights_sum <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights_sum

    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=max_points, replace=False, p=weights)

    return x[idx], y[idx], z[idx], field[idx]


# -------------------------------------------------
# STREAMLIT SETTINGS
# -------------------------------------------------
#st.set_page_config(
#    page_title="Thermal Digital Twin for TOFAS Pilot",
#    layout="wide"
#)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { margin-top: 0rem; padding-top: 0rem; }
    h1 { margin-top: 0rem; padding-top: 0rem; }
    .element-container { margin-top: 0rem; }

    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div { background: #dbeafe; }
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background: #2563eb; }
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
        background: #ffffff;
        border: 2px solid #2563eb;
        box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.35);
    }

    .sidebar-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
        border: 1px solid #e5e7eb;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 13px;
    }
    .sidebar-card-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }
    .sidebar-card-metric {
        font-size: 13px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Thermal Twin for TOFAS Pilot")

npz_path = "TOFAS/TOFAScase.npz"
vtk_path = "TOFAS/TOFAScase.vtk"

mesh, T_field, U_field = load_npz_case(npz_path, vtk_path)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.markdown(
    """
    <h2 style='text-align: center; font-size:20px; margin-bottom: 0.2rem;'>
        CFD-Enabled Digital Twin for Data Centres
    </h2>
    <hr>
    """,
    unsafe_allow_html=True
)

view_tab = st.sidebar.radio(" ", ["Digital Twin", "About"], index=0)

# Initialize defaults
field_choice = "Temperature"
mode = "3D Scatter"
slice_axis = "Z"
thickness_percent = 3
grid_resolution = 1000
max_points = 3000
value_min = 0.0
value_max = 1.0
x = y = z = None
field = None
color_label = ""
total_cells = 0

# Only show settings when in Digital Twin tab
if view_tab == "Digital Twin":
    
    # 📂 Visualization Settings
    with st.sidebar.expander("📂 Visualization Settings", expanded=False):
        field_choice = st.selectbox("Field to Display", ["Temperature", "Airflow Velocity"])
        mode = st.selectbox("View Mode", ["3D Scatter", "2D Slice"])
    
    # Single source of truth: field + slider only here
    x, y, z, field, color_label = get_coords_and_field(mesh, T_field, U_field, field_choice)
    total_cells = x.size

    # Calculate field-specific limits
    if field_choice == "Temperature":
        field_min = float(field.min())
        field_max = 30.10
        low_default = field_min
        high_default = field_max
    else:
        field_min = float(field.min())
        field_max = float(1.48)
        low_default = field_min
        high_default = field_max
    
    # 🔍 Filters & Slicing
    with st.sidebar.expander("🔍 Filters & Slicing", expanded=False):
        # Range slider
        value_min, value_max = st.slider(
            f"{color_label} Range Filter",
            min_value=field_min,
            max_value=field_max,
            value=(low_default, high_default),
            help=f"Only show points between {color_label}"
        )
        
        # Maximum points slider
        max_points = st.slider(
            "Maximum Points (3D only)",
            min_value=1000,
            max_value=5000,
            value=3000,
            step=500,
            help="More points = slower performance"
        )
        
        # 2D Slice specific settings
        if mode == "2D Slice":
            slice_axis = st.selectbox("Slice Axis", ["X", "Y", "Z"])
            
            thickness_percent = st.slider(
                "Slice Thickness (%)",
                1, 5, 3, 1,
                help="Thicker slice = more data points but less precise location"
            )
            
            grid_resolution = st.slider(
                "Grid Resolution",
                500, 2000, 1000, 100,
                help="Higher = sharper but slower. 700-800 recommended."
            )
    
    # ℹ️ Mesh Info
    with st.sidebar.expander("ℹ️ Mesh Info", expanded=False):
        displayed_points = min(total_cells, max_points)
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-card-title">Grid Size</div>
                <div class="sidebar-card-metric">Total cells: {total_cells:,}</div>
                <div class="sidebar-card-metric">Displayed: {displayed_points:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.sidebar.markdown(
    """
    <p style='font-size: 12px; color:#6b7280; font-weight:600; margin-bottom:4px;'>
        Developed by D&STECH © 2026
    </p>
    <p style='font-size: 11px; color:#6b7280; line-height:1.4;'>
        FluxTwin(c) has been developed as part of EU-funded HEATWISE Project. The Heatwise Project has received funding from the European Union's Horizon Europe research and innovation programme under Grant Agreement No 101138491 and the Swiss Secretariat for Education, Research, and Innovation (SERI) under contract No 23.00606.
    </p>
    """,
    unsafe_allow_html=True
)

logo_col1, logo_col2 = st.sidebar.columns(2)

heatwise_src = file_to_data_uri(BASE_DIR / "heatwise_logo.svg", "image/svg+xml")
dstech_src   = file_to_data_uri(BASE_DIR / "dstech_logo.png", "image/png")

with logo_col1:
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


# -------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------
if view_tab == "About":
    st.image("TOFAS/tofas.png", caption="Tofaş Türk Otomobil Fabrikası A.Ş.", width=400)

    st.markdown("""
### Facility Information
- **Location:** Bursa, Türkiye
- **Cooling Technology:** Air cooling infrastructure

### IT Infrastructure
- **Number of Server Racks:** 43 Racks
- **Total IT Capacity:** 19 kW
- **Rack Power Range:** 852 W - 5848 W
- **Configuration:** Variable thermal loads

### Cooling System
- **Cooling Method:** Air-based cooling units
- **Number of Cooling Units:** 4 (2 active)

### Operating Conditions
- **Ambient Temperature:** 17°C
- **Inlet Air Temperature:** 17.5°C
- **Outlet Air Temperature:** 25°C
""")

    st.markdown("""
---
<h3 style='font-style: italic;'>D&S Tech | Digital Twin Solutions</h3>
<p style='font-size: 14px;'><a href='https://dstechs.net/' target='_blank'>https://dstechs.net/</a></p>
<p style='font-size: 16px;'>Get Your Thermal Digital Twin. Contact us today: <a href='mailto:datacenter@dstechs.net'><strong>datacenter@dstechs.net</strong></a></p>
""", unsafe_allow_html=True)

    st.stop()
# -------------------------------------------------
# RESULTS PAGE
# -------------------------------------------------
elif view_tab == "Digital Twin":
    col1, col2 = st.columns([7, 3])

    with col1:
        viz_title = "🌡️ Data Centre Thermal Map" if field_choice == "Temperature" else "💨 Airflow Velocity Field"
        st.subheader(viz_title)

        # Downsampling (3D)
        x_plot, y_plot, z_plot, f_plot = density_aware_downsample(x, y, z, field, max_points)

        # Apply range filter
        mask_range = (f_plot >= value_min) & (f_plot <= value_max)
        x_plot = x_plot[mask_range]
        y_plot = y_plot[mask_range]
        z_plot = z_plot[mask_range]
        f_plot = f_plot[mask_range]

        # Colorbar = slider range
        cmin = float(value_min)
        cmax = float(value_max)

        # -----------------------------------------
        # 3D SCATTER
        # -----------------------------------------
        if mode == "3D Scatter":
            fig = go.Figure()

            # Outer geometry
            try:
                gx, gy, gz, gi, gj, gk = load_outer_geometry(vtk_path)
                fig.add_trace(go.Mesh3d(
                    x=gx, y=gy, z=gz,
                    i=gi, j=gj, k=gk,
                    opacity=0.25,
                    color="gray",
                    name="DC Geometry",
                    showscale=False,
                    lighting=dict(
                        ambient=1.0, diffuse=0.0, specular=0.0,
                        roughness=0.0, fresnel=0.0
                    ),
                    hoverinfo="skip"
                ))
            except Exception as e:
                st.warning(f"Outer geometry (VTK) could not be loaded: {e}")

            fig.add_trace(go.Scatter3d(
                x=x_plot, y=y_plot, z=z_plot,
                mode="markers",
                marker=dict(
                    size=1.6,
                    color=f_plot,
                    colorscale="Turbo",
                    cmin=cmin, cmax=cmax,
                    opacity=0.7,
                    colorbar=dict(
                        title=dict(text=color_label, font=dict(color="black", size=14)),
                        tickfont=dict(color="black", size=12),
                    ),
                ),
                hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<br>"
                              + color_label + ": %{marker.color:.3f}<extra></extra>"
            ))

            fig.update_layout(
                height=700,
                scene=dict(
                    xaxis_title="X [m]",
                    yaxis_title="Y [m]",
                    zaxis_title="Z [m]",
                    bgcolor="white",
                    aspectmode="data",
                ),
                scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="white",
            )

            st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------
        # 2D SLICE (PLANE IN 3D)
        # -----------------------------------------
        else:
            x_min, x_max = float(x.min()), float(x.max())
            y_min, y_max = float(y.min()), float(y.max())
            z_min, z_max = float(z.min()), float(z.max())

            if slice_axis == "X":
                coord = x
                coord_min, coord_max = x_min, x_max
                axis1, axis2 = y, z
                axis1_label, axis2_label = "Y [m]", "Z [m]"
            elif slice_axis == "Y":
                coord = y
                coord_min, coord_max = y_min, y_max
                axis1, axis2 = x, z
                axis1_label, axis2_label = "X [m]", "Z [m]"
            else:
                coord = z
                coord_min, coord_max = z_min, z_max
                axis1, axis2 = x, y
                axis1_label, axis2_label = "X [m]", "Y [m]"

            default_coord = 0.5 * (coord_min + coord_max)

            slice_coord = st.slider(
                f"{slice_axis}-slice location",
                min_value=coord_min,
                max_value=coord_max,
                value=default_coord,
                step=(coord_max - coord_min) / 200,
            )

            thickness = (thickness_percent / 100.0) * (coord_max - coord_min)
            mask = np.abs(coord - slice_coord) <= thickness

            axis1_s = axis1[mask]
            axis2_s = axis2[mask]
            f_s = field[mask]

            if axis1_s.size == 0:
                st.warning(f"No points at this {slice_axis} location, adjust the slice.")
            else:
                st.caption(
                    f"{axis1_s.size:,} points in slice "
                    f"(thickness: {thickness_percent}%) → "
                    f"Interpolating to {grid_resolution}x{grid_resolution} grid"
                )

                try:
                    grid_axis1, grid_axis2, grid_field = interpolate_slice(axis1_s, axis2_s, f_s, grid_resolution)

                    grid_field_filtered = grid_field.copy()
                    grid_field_filtered[(grid_field_filtered < value_min) | (grid_field_filtered > value_max)] = np.nan

                    A1, A2 = np.meshgrid(grid_axis1, grid_axis2)

                    if slice_axis == "X":
                        X_plane = np.full_like(A1, slice_coord)
                        Y_plane = A1
                        Z_plane = A2
                    elif slice_axis == "Y":
                        Y_plane = np.full_like(A1, slice_coord)
                        X_plane = A1
                        Z_plane = A2
                    else:
                        Z_plane = np.full_like(A1, slice_coord)
                        X_plane = A1
                        Y_plane = A2

                    fig_slice3d = go.Figure()

                    try:
                        gx, gy, gz, gi, gj, gk = load_outer_geometry(vtk_path)
                        fig_slice3d.add_trace(go.Mesh3d(
                            x=gx, y=gy, z=gz,
                            i=gi, j=gj, k=gk,
                            opacity=0.15,
                            color="gray",
                            name="DC Geometry",
                            showscale=False,
                            hoverinfo="skip"
                        ))
                    except Exception as e:
                        st.warning(f"Outer geometry (VTK) could not be loaded: {e}")

                    fig_slice3d.add_trace(go.Surface(
                        x=X_plane, y=Y_plane, z=Z_plane,
                        surfacecolor=grid_field_filtered,
                        colorscale="Turbo",
                        cmin=value_min, cmax=value_max,
                        colorbar=dict(
                            title=dict(text=color_label, font=dict(color="black", size=14)),
                            tickfont=dict(color="black", size=12),
                        ),
                        opacity=0.9,
                        name="Slice"
                    ))

                    fig_slice3d.update_layout(
                        height=700,
                        margin=dict(l=0, r=0, t=40, b=0),
                        scene=dict(
                            xaxis_title="X [m]",
                            yaxis_title="Y [m]",
                            zaxis_title="Z [m]",
                            bgcolor="white",
                            aspectmode="data",
                        ),
                        paper_bgcolor="white",
                    )

                    st.plotly_chart(fig_slice3d, use_container_width=True)

                except Exception as e:
                    st.error(f"Slice interpolation failed: {e}")

    # -------------------------------------------------
    # STATS - DYNAMIC TITLES AND UNITS
    # -------------------------------------------------
    with col2:
        # Title and metrics based on field selection
        if field_choice == "Temperature":
            analysis_title = "🌡️ Global Thermal Analysis"
            unit = "°C"
            metric_names = {
                "min": "🌐 Facility Minimum Temperature",
                "avg": "📊 Facility Average Temperature",
                "max": "🔥 Facility Peak Temperature",
                "std": "📉 Thermal Uniformity (Std.Dev)"
            }
            # Peak Temperature for TOFAS: value_max (slider value)
            peak_value = value_max
        else:  # Airflow Velocity
            analysis_title = "💨 Airflow Analysis"
            unit = "m/s"
            metric_names = {
                "min": "🌐 Minimum Air Velocity",
                "avg": "📊 Average Air Velocity",
                "max": "🔥 Peak Air Velocity",
                "std": "📉 Airflow Uniformity (Std.Dev)"
            }
            peak_value = value_max
        
        st.markdown(f"<h3 style='font-size: 20px; font-weight: bold;'>{analysis_title}</h3>", unsafe_allow_html=True)
        
        st.write(f"{metric_names['min']} = {field.min():.2f} {unit}")
        st.write(f"{metric_names['avg']} = {field.mean():.2f} {unit}")
        st.write(f"{metric_names['max']} = {peak_value:.2f} {unit}")
        st.write(f"{metric_names['std']} = {field.std():.2f} {unit}")

        st.markdown("---")
        st.markdown("""
        <div class="powered-by" style="margin-top: 8px; font-weight: bold; font-size: 20px;">
            Powered by D&S Tech
        </div>
        <div class="website-link" style="margin-top: 0px; font-size: 13px;">
            <a href="https://dstechs.net/" target="_blank">https://dstechs.net/</a>
        </div>
        """, unsafe_allow_html=True)
