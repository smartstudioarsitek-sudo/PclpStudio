import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from shapely.geometry import Polygon, LineString

# --- HANDLING IMPORT LIBRARY ---
try:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
except ImportError:
    st.warning("⚠️ Library 'ezdxf' belum terinstall. Fitur DXF tidak akan jalan.")

HAS_GEO_LIBS = False
try:
    import geopandas as gpd
    import rasterio
    from rasterio.plot import show
    HAS_GEO_LIBS = True
except ImportError:
    pass

# ==========================================
# 1. PARSER ENGINE
# ==========================================
def parse_pclp_block(df):
    """Parser untuk format Excel Blok PCLP (Cross Section)."""
    parsed_data = []
    i = 0
    df = df.astype(str)
    
    while i < len(df):
        row = df.iloc[i].values
        x_indices = [idx for idx, val in enumerate(row) if val.strip().upper() == 'X']
        
        if x_indices and (i + 1 < len(df)):
            x_idx = x_indices[0]
            val_y = str(df.iloc[i+1, x_idx]).strip().upper()
            
            if val_y == 'Y':
                sta_name = f"STA_{len(parsed_data)}"
                candidate_sta = str(df.iloc[i+1, 1]).strip() 
                if candidate_sta.lower() not in ['nan', 'none', '']:
                    sta_name = candidate_sta
                if sta_name.endswith('.0'): sta_name = sta_name[:-2]

                start_col = x_idx + 1
                row_x = df.iloc[i].values
                row_y = df.iloc[i+1].values
                points = []
                for c in range(start_col, len(row_x)):
                    try:
                        vx = float(str(row_x[c]).replace(',', '.'))
                        vy = float(str(row_y[c]).replace(',', '.'))
                        if not (math.isnan(vx) or math.isnan(vy)):
                            points.append((vx, vy))
                    except: break
                
                if points:
                    points.sort(key=lambda p: p[0])
                    parsed_data.append({'STA': sta_name, 'points': points})
                i += 1
        i += 1
    return parsed_data

def hitung_cut_fill(tanah_pts, desain_pts):
    if not tanah_pts or not desain_pts: return 0.0, 0.0
    min_y = min([p[1] for p in tanah_pts] + [p[1] for p in desain_pts])
    datum = min_y - 5.0
    p_tanah = tanah_pts + [(tanah_pts[-1][0], datum), (tanah_pts[0][0], datum)]
    p_desain = desain_pts + [(desain_pts[-1][0], datum), (desain_pts[0][0], datum)]
    poly_tanah = Polygon(p_tanah).buffer(0)
    poly_desain = Polygon(p_desain).buffer(0)
    try:
        return poly_desain.intersection(poly_tanah).area, poly_desain.difference(poly_tanah).area
    except: return 0.0, 0.0

# ==========================================
# 2. GENERATOR OUTPUT (STANDAR KP IRIGASI)
# ==========================================
def generate_dxf(results, mode="cross"):
    """
    Generate DXF dengan Standar KP (Irigasi/Bina Marga).
    Fixed: Menggunakan 'lineweight' bukan 'linewidth'.
    """
    doc = ezdxf.new('R2010')
    
    # --- SETUP LINETYPES (MANUAL DEFINITION) ---
    if 'DASHED' not in doc.linetypes:
        doc.linetypes.new('DASHED', dxfattribs={'description': 'Dashed', 'pattern': [0.75, 0.5, -0.25]})
    if 'CENTER' not in doc.linetypes: 
        doc.linetypes.new('CENTER', dxfattribs={'description': 'Center', 'pattern': [1.25, 0.25, -0.25, 0.25]})

    msp = doc.modelspace()

    # --- SETUP LAYERS ---
    # FIX: linewidth -> lineweight (AutoCAD Standard)
    doc.layers.add(name='TANAH_ASLI', color=8, linetype='DASHED') 
    doc.layers.add(name='DESAIN_RENCANA', color=1, lineweight=30)  
    doc.layers.add(name='TEXT_DATA', color=2)      # Kuning
    doc.layers.add(name='TEXT_LABEL', color=7)     # Putih
    doc.layers.add(name='GRID_MAJOR', color=9, linetype='CENTER') 
    doc.layers.add(name='FRAME_TABLE', color=7)    
    doc.layers.add(name='KOP_GAMBAR', color=3)     

    # --- KONSTANTA SKALA & UKURAN ---
    SC_H = 1.0   # Skala Horizontal 1:100
    SC_V = 10.0  # Skala Vertikal 1:10 (Exaggerated)
    ROW_H = 15.0 # Tinggi per baris tabel
    
    # Style Text
    if "ARIAL" not in doc.styles:
        doc.styles.new("ARIAL", dxfattribs={'font': 'Arial.ttf'})
    if "ARIAL_NARROW" not in doc.styles:
        doc.styles.new("ARIAL_NARROW", dxfattribs={'font': 'Arial Narrow.ttf'})

    def draw_kp_profile(origin_x, origin_y, points_tanah, points_desain, sta_title):
        """Menggambar 1 Profile (Cross/Long) lengkap dengan Tabel Data KP."""
        all_pts = points_tanah + points_desain
        if not all_pts: return 0, 0

        min_x = min(p[0] for p in all_pts)
        max_x = max(p[0] for p in all_pts)
        min_y = min(p[1] for p in all_pts)
        max_y = max(p[1] for p in all_pts)

        # Rounding Grid
        g_min_x = math.floor(min_x / 2.0) * 2.0 
        g_max_x = math.ceil(max_x / 2.0) * 2.0
        g_min_y = math.floor(min_y / 1.0) * 1.0
        g_max_y = math.ceil(max_y / 1.0) * 1.0
        
        datum_graph = g_min_y 
        graph_w = (g_max_x - g_min_x) * SC_H
        graph_h = (g_max_y - g_min_y) * SC_V
        
        # Posisi Awal Grafik (Di atas tabel data)
        TABLE_OFFSET_Y = 3 * ROW_H 
        base_graph_x = origin_x
        base_graph_y = origin_y + TABLE_OFFSET_Y

        # --- A. GAMBAR GRID & DATA VERTIKAL ---
        curr_x = g_min_x
        while curr_x <= g_max_x + 0.01:
            draw_x = base_graph_x + (curr_x - g_min_x) * SC_H
            
            # 1. Garis Grid Vertikal
            msp.add_line((draw_x, base_graph_y + graph_h), (draw_x, origin_y), dxfattribs={'layer': 'GRID_MAJOR'})
            
            # 2. Interpolasi Elevasi
            def get_elev(pts, x_val):
                for k in range(len(pts)-1):
                    p1, p2 = pts[k], pts[k+1]
                    if p1[0] <= x_val <= p2[0]:
                        ratio = (x_val - p1[0]) / (p2[0] - p1[0]) if (p2[0]-p1[0]) !=0 else 0
                        return p1[1] + ratio * (p2[1] - p1[1])
                return None

            z_tanah = get_elev(points_tanah, curr_x)
            z_desain = get_elev(points_desain, curr_x)

            # 3. Tulis Angka di Tabel (Rotasi 90)
            txt_dist = msp.add_text(f"{curr_x:.1f}", dxfattribs={'height': 1.8, 'layer': 'TEXT_DATA', 'style': 'ARIAL_NARROW', 'rotation': 90})
            txt_dist.set_placement((draw_x + 1, origin_y + (0.5 * ROW_H)), align=TextEntityAlignment.MIDDLE_CENTER)
            
            if z_tanah is not None:
                txt_t = msp.add_text(f"{z_tanah:.2f}", dxfattribs={'height': 1.8, 'layer': 'TEXT_DATA', 'style': 'ARIAL_NARROW', 'rotation': 90})
                txt_t.set_placement((draw_x + 1, origin_y + (1.5 * ROW_H)), align=TextEntityAlignment.MIDDLE_CENTER)
            
            if z_desain is not None:
                txt_d = msp.add_text(f"{z_desain:.2f}", dxfattribs={'height': 1.8, 'layer': 'TEXT_DATA', 'style': 'ARIAL_NARROW', 'rotation': 90})
                txt_d.set_placement((draw_x + 1, origin_y + (2.5 * ROW_H)), align=TextEntityAlignment.MIDDLE_CENTER)

            curr_x += 2.0 
            
        # --- B. GAMBAR GARIS DATA (POLYLINE) ---
        if points_tanah:
            p_draw = [(base_graph_x + (p[0]-g_min_x)*SC_H, base_graph_y + (p[1]-datum_graph)*SC_V) for p in points_tanah]
            msp.add_lwpolyline(p_draw, dxfattribs={'layer': 'TANAH_ASLI'})
            
        if points_desain:
            p_draw = [(base_graph_x + (p[0]-g_min_x)*SC_H, base_graph_y + (p[1]-datum_graph)*SC_V) for p in points_desain]
            msp.add_lwpolyline(p_draw, dxfattribs={'layer': 'DESAIN_RENCANA'})

        # --- C. FRAME & LABEL BARIS ---
        width_tot = graph_w
        for i in range(4):
            y_line = origin_y + (i * ROW_H)
            msp.add_line((origin_x, y_line), (origin_x + width_tot, y_line), dxfattribs={'layer': 'FRAME_TABLE'})
        
        msp.add_line((origin_x, base_graph_y + graph_h), (origin_x + width_tot, base_graph_y + graph_h), dxfattribs={'layer': 'FRAME_TABLE'})
        msp.add_line((origin_x, origin_y), (origin_x, base_graph_y + graph_h), dxfattribs={'layer': 'FRAME_TABLE'})
        msp.add_line((origin_x + width_tot, origin_y), (origin_x + width_tot, base_graph_y + graph_h), dxfattribs={'layer': 'FRAME_TABLE'})

        offset_lbl = -2.0
        msp.add_text("JARAK", dxfattribs={'height': 2.0, 'layer': 'TEXT_LABEL', 'style': 'ARIAL'}).set_placement((origin_x + offset_lbl, origin_y + 0.5*ROW_H), align=TextEntityAlignment.MIDDLE_RIGHT)
        msp.add_text("ELEV. TANAH", dxfattribs={'height': 2.0, 'layer': 'TEXT_LABEL', 'style': 'ARIAL'}).set_placement((origin_x + offset_lbl, origin_y + 1.5*ROW_H), align=TextEntityAlignment.MIDDLE_RIGHT)
        msp.add_text("ELEV. DESAIN", dxfattribs={'height': 2.0, 'layer': 'TEXT_LABEL', 'style': 'ARIAL'}).set_placement((origin_x + offset_lbl, origin_y + 2.5*ROW_H), align=TextEntityAlignment.MIDDLE_RIGHT)
        
        curr_y = g_min_y
        while curr_y <= g_max_y:
            y_pos = base_graph_y + (curr_y - g_min_y) * SC_V
            msp.add_line((origin_x, y_pos), (origin_x + width_tot, y_pos), dxfattribs={'layer': 'GRID_MAJOR'})
            msp.add_text(f"{curr_y:.2f}", dxfattribs={'height': 2.0, 'layer': 'TEXT_LABEL'}).set_placement((origin_x - 1, y_pos), align=TextEntityAlignment.MIDDLE_RIGHT)
            curr_y += 1.0

        msp.add_text(sta_title, dxfattribs={'height': 4.0, 'layer': 'TEXT_LABEL', 'style': 'ARIAL'}).set_placement((origin_x + width_tot/2, base_graph_y + graph_h + 5), align=TextEntityAlignment.CENTER)
        msp.add_text(f"DATUM {datum_graph:.2f}", dxfattribs={'height': 2.5, 'layer': 'TEXT_LABEL'}).set_placement((origin_x - 5, base_graph_y), align=TextEntityAlignment.MIDDLE_RIGHT)

        return graph_w, graph_h + TABLE_OFFSET_Y 

    # --- MAIN LOOP ---
    if mode == "long":
        tanah, desain = results
        draw_kp_profile(0, 0, tanah, desain, "LONG SECTION PROFILE")
    else:
        curr_x = 0
        curr_y = 0
        max_h_row = 0
        
        for item in results:
            w, h = draw_kp_profile(curr_x, curr_y, item.get('points_tanah', []), item.get('points_desain', []), item['STA'])
            curr_x += w + 50 # Spasi antar gambar
            max_h_row = max(max_h_row, h)
            
            if curr_x > 500: # Ganti baris jika terlalu lebar
                curr_x = 0
                curr_y -= (max_h_row + 50) 
                max_h_row = 0

    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode('utf-8')

def generate_excel_report(data):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        rekap = []
        for item in data:
            rekap.append({'STA': item['STA'], 'Cut Area (m2)': item['cut'], 'Fill Area (m2)': item['fill']})
        pd.DataFrame(rekap).to_excel(writer, sheet_name='Volume Report', index=False)
    return output.getvalue()

# ==========================================
# 3. GEOSPATIAL ENGINE
# ==========================================
def extract_long_section_from_dem(dem_file, shp_file, interval=25):
    if not HAS_GEO_LIBS: return None, "Library GIS Missing"
    try:
        with rasterio.open(dem_file) as src:
            gdf = gpd.read_file(shp_file)
            if gdf.crs != src.crs: gdf = gdf.to_crs(src.crs)
            line = gdf.geometry.iloc[0]
            if line.geom_type == 'MultiLineString': line = line.geoms[0]
            
            length = line.length
            points_data = []
            for dist in np.arange(0, length, interval):
                pt = line.interpolate(dist)
                try:
                    for val in src.sample([(pt.x, pt.y)]):
                        elev = val[0]
                        if elev == src.nodata: elev = np.nan
                        points_data.append({'Station (m)': dist, 'Elevation (m)': elev, 'X': pt.x, 'Y': pt.y})
                except: pass
            return pd.DataFrame(points_data), None
    except Exception as e: return None, str(e)

def extract_cross_section_from_dem(dem_file, shp_file, interval=50, width_left=25, width_right=25, step=1.0):
    if not HAS_GEO_LIBS: return None, None, "Library GIS Missing"
    cross_data_app = [] 
    cross_data_civil = [] 
    try:
        with rasterio.open(dem_file) as src:
            gdf = gpd.read_file(shp_file)
            if gdf.crs != src.crs: gdf = gdf.to_crs(src.crs)
            line = gdf.geometry.iloc[0]
            if line.geom_type == 'MultiLineString': line = line.geoms[0]
            length = line.length
            for dist in np.arange(0, length + 0.1, interval):
                pt_center = line.interpolate(dist)
                p_back = line.interpolate(max(0, dist - 0.1))
                p_front = line.interpolate(min(length, dist + 0.1))
                dx = p_front.x - p_back.x
                dy = p_front.y - p_back.y
                len_v = math.sqrt(dx**2 + dy**2)
                if len_v == 0: continue
                nx, ny = -dy/len_v, dx/len_v
                offsets = np.arange(-width_left, width_right + 0.1, step)
                points_tanah = []
                for offset in offsets:
                    sample_x = pt_center.x + (nx * offset)
                    sample_y = pt_center.y + (ny * offset)
                    elev = np.nan
                    try:
                        for val in src.sample([(sample_x, sample_y)]):
                            elev = val[0]
                            if elev == src.nodata: elev = np.nan
                    except: pass
                    if not np.isnan(elev):
                        points_tanah.append((offset, elev))
                        cross_data_civil.append({'Station': dist, 'Offset': offset, 'Elevation': elev, 'Easting': sample_x, 'Northing': sample_y})
                if points_tanah:
                    cross_data_app.append({'STA': f"STA {int(dist)}+00", 'points_tanah': points_tanah, 'points_desain': [], 'cut': 0.0, 'fill': 0.0})
        return cross_data_app, pd.DataFrame(cross_data_civil), None
    except Exception as e: return None, None, str(e)

def render_peta_situasi(dem_file, shp_file):
    if not HAS_GEO_LIBS: return None, "No GIS Libs"
    try:
        with rasterio.open(dem_file) as src:
            gdf = gpd.read_file(shp_file)
            if gdf.crs != src.crs: gdf = gdf.to_crs(src.crs)
            fig, ax = plt.subplots(figsize=(10, 8))
            data = src.read(1, out_shape=(src.height//5, src.width//5))
            data_masked = np.ma.masked_where(data == src.nodata, data)
            x = np.linspace(src.bounds.left, src.bounds.right, data.shape[1])
            y = np.linspace(src.bounds.top, src.bounds.bottom, data.shape[0])
            X, Y = np.meshgrid(x, y)
            contours = ax.contour(X, Y, data_masked, levels=20, cmap='terrain', linewidths=0.5)
            ax.clabel(contours, inline=True, fontsize=6, fmt='%1.0f')
            gdf.plot(ax=ax, color='red', linewidth=2, label='Trase', zorder=5)
            ax.grid(True, linestyle='--', alpha=0.5); ax.set_title("Peta Situasi")
            return fig, None
    except Exception as e: return None, str(e)

# ==========================================
# 4. MAIN UI
# ==========================================
st.set_page_config(page_title="PCLP Studio", layout="wide")
st.title("🚜 PCLP Studio Pro v6.1")
st.caption("Aplikasi Desain Irigasi & Jalan: Cross Section, Long Section & GIS Situasi")

if not HAS_GEO_LIBS: st.warning("⚠️ Modul Geospasial tidak aktif.")

# --- MENYUSUN TAB SESUAI URUTAN BARU ---
tabs = st.tabs(["📖 MANUAL BOOK", "🗺️ PETA SITUASI (GIS)", "📈 LONG SECTION", "📐 CROSS SECTION"])

# --- TAB 1: MANUAL BOOK ---
with tabs[0]:
    st.markdown("""
    ## 📖 Panduan Penggunaan Aplikasi
    Selamat datang di **PCLP Studio Pro**. Aplikasi ini membantu insinyur sipil untuk mengolah data pengukuran tanah, 
    menghitung volume cut & fill, serta ekstraksi data topografi otomatis.
    
    ---
    ### 1. Fitur GIS (Peta Situasi)
    Mulai dari sini jika Anda menggunakan data satelit/drone (DEM & SHP).
    1. Upload **DEM (.tif)** dan **Trase (.shp/.geojson)**.
    2. Ekstrak data **Long Section** & **Cross Section** secara otomatis.
    3. Hasil ekstraksi akan otomatis dikirim ke Tab Long & Cross.
    
    ---
    ### 2. Fitur Cross Section (Manual/Auto)
    Bisa menerima data manual (Excel) atau hasil dari GIS.
    * **Manual:** Upload Excel format PCLP.
    * **Auto:** Data otomatis muncul jika diekstrak dari Tab GIS.
    * Output: DXF Standar KP Irigasi.
    """)

# --- TAB 2: PETA SITUASI (GIS) ---
# Dulu Tab 4, sekarang jadi Tab 2
with tabs[1]:
    st.header("🗺️ Peta Situasi & Ekstraksi Data")
    c1, c2 = st.columns([1, 2])
    with c1:
        up_dem = st.file_uploader("Upload DEM (.tif)", type=['tif', 'tiff'])
        up_shp = st.file_uploader("Upload Trase (.geojson/.shp)", type=['geojson', 'shp'], accept_multiple_files=True)
        st.markdown("---")
        interval = st.number_input("Interval Antar STA (m)", 5, 1000, 25, 5)
        col_w1, col_w2 = st.columns(2)
        w_left = col_w1.number_input("Lebar Kiri (m)", 5, 100, 25, 5)
        w_right = col_w2.number_input("Lebar Kanan (m)", 5, 100, 25, 5)
        shp_file = None
        if up_shp:
            for f in up_shp:
                if f.name.endswith('.geojson') or f.name.endswith('.shp'): shp_file = f; break
        btn_render = st.button("1. Tampilkan Peta")
        btn_long = st.button("2. Ekstrak Long Section")
        btn_cross = st.button("3. Ekstrak Cross Section (Auto)")
    with c2:
        if btn_render and up_dem and shp_file:
            st.session_state['gis_files'] = (up_dem, shp_file)
        if 'gis_files' in st.session_state:
            dem, shp = st.session_state['gis_files']
            dem.seek(0); shp.seek(0)
            with st.spinner("Merender Peta..."):
                fig, err = render_peta_situasi(dem, shp)
                if fig: st.pyplot(fig)
        if btn_long and up_dem and shp_file:
            up_dem.seek(0); shp_file.seek(0)
            with st.spinner(f"Extracting Long Section ({interval}m)..."):
                df_long, err = extract_long_section_from_dem(up_dem, shp_file, interval)
                if df_long is not None:
                    st.success(f"✅ Long Section: {len(df_long)} titik")
                    long_data = df_long[['Station (m)', 'Elevation (m)']].dropna().values.tolist()
                    st.session_state['long_res'] = (long_data, [])
                    st.info("Data berhasil dikirim ke Tab 'LONG SECTION'. Silakan buka tab sebelah.")
                    st.download_button("📥 Download Excel Long", io.BytesIO(b""), "Long_Section.xlsx")
        if btn_cross and up_dem and shp_file:
            up_dem.seek(0); shp_file.seek(0)
            with st.spinner(f"Generating Cross Sections..."):
                app_data, df_civil, err = extract_cross_section_from_dem(up_dem, shp_file, interval, w_left, w_right)
                if app_data:
                    st.session_state['data_cross'] = app_data
                    st.success(f"✅ Berhasil: {len(app_data)} Cross Section!")
                    st.info("Data berhasil dikirim ke Tab 'CROSS SECTION'. Silakan buka tab paling kanan.")
                    dxf_bytes = generate_dxf(app_data, "cross")
                    st.download_button("📥 Download DXF (Std KP)", dxf_bytes, "Cross_Section_KP.dxf")

# --- TAB 3: LONG SECTION ---
with tabs[2]:
    st.subheader("Long Section")
    f_long = st.file_uploader("Upload Long", type=['xls', 'xlsx', 'csv'], key='long_up')
    if f_long:
        try:
            df = pd.read_csv(f_long) if f_long.name.endswith('.csv') else pd.read_excel(f_long)
            st.session_state['long_res'] = (df.iloc[:, :2].dropna().values.tolist(), [])
            st.success("Data masuk!")
        except: st.error("Error file.")
    if 'long_res' in st.session_state:
        ogl, _ = st.session_state['long_res']
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(*zip(*ogl), 'k--', label='Tanah Asli')
        ax.grid(True); st.pyplot(fig)
        st.download_button("📥 DXF Long (Std KP)", generate_dxf((ogl, []), "long"), "Long_KP.dxf")

# --- TAB 4: CROSS SECTION ---
with tabs[3]:
    col_in, col_view = st.columns([1, 2])
    with col_in:
        st.subheader("Input Data PCLP (Manual)")
        f_upload = st.file_uploader("Upload Excel", type=['xls', 'xlsx'], key='cross_up')
        if f_upload:
            try:
                xls = pd.ExcelFile(f_upload)
                s_ogl = st.selectbox("Sheet Tanah", ["[Pilih]"]+xls.sheet_names)
                s_dsn = st.selectbox("Sheet Desain", ["[Pilih]"]+xls.sheet_names)
                if st.button("PROSES DATA"):
                    d_ogl = parse_pclp_block(pd.read_excel(f_upload, sheet_name=s_ogl, header=None)) if s_ogl != "[Pilih]" else []
                    d_dsn = parse_pclp_block(pd.read_excel(f_upload, sheet_name=s_dsn, header=None)) if s_dsn != "[Pilih]" else []
                    final = []
                    for i in range(max(len(d_ogl), len(d_dsn))):
                        t = d_ogl[i] if i < len(d_ogl) else None
                        d = d_dsn[i] if i < len(d_dsn) else None
                        sta = t['STA'] if t else (d['STA'] if d else f"STA_{i}")
                        tp, dp = (t['points'] if t else []), (d['points'] if d else [])
                        c, f = hitung_cut_fill(tp, dp)
                        final.append({'STA': sta, 'points_tanah': tp, 'points_desain': dp, 'cut': c, 'fill': f})
                    st.session_state['data_cross'] = final
                    st.success("Selesai!")
            except: st.error("Gagal baca file.")

    with col_view:
        if 'data_cross' in st.session_state:
            data = st.session_state['data_cross']
            idx = st.slider("Pilih STA", 0, len(data)-1, 0)
            item = data[idx]
            fig, ax = plt.subplots(figsize=(10, 4))
            if item['points_tanah']: ax.plot(*zip(*item['points_tanah']), 'k-o', label='Tanah')
            if item['points_desain']: ax.plot(*zip(*item['points_desain']), 'r-', label='Desain')
            ax.set_title(f"{item['STA']} | C:{item['cut']:.2f} | F:{item['fill']:.2f}")
            ax.legend(); ax.grid(True); st.pyplot(fig)
            c1, c2 = st.columns(2)
            c1.download_button("📥 DXF Cross (Std KP)", generate_dxf(data, "cross"), "Cross_KP.dxf")
            c2.download_button("📥 Excel Report", generate_excel_report(data), "Vol_Report.xlsx")
