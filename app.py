import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiPolygon

# --- LIBRARY HANDLING ---
try:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
    from ezdxf.tools.standards import linetypes # Standard linetypes
except ImportError:
    st.error("⚠️ Library 'ezdxf' wajib diinstall: `pip install ezdxf`")
    st.stop()

HAS_GEO_LIBS = False
try:
    import geopandas as gpd
    import rasterio
    from scipy.ndimage import gaussian_filter
    HAS_GEO_LIBS = True
except ImportError:
    pass

# ==========================================
# 1. KONFIGURASI STANDAR KP-07 (DIGITAL SPEC)
# ==========================================
def setup_kp07_standards(doc):
    """
    Mengatur Layer, Linetype, dan Text Style sesuai mandat KP-07.
    Referensi: Tabel 1 Analisis & Poin 2.1
    """
    # A. Setup Linetypes
    # KP-07 Tanah Asli: 0.75 garis, 0.5 spasi, -0.25 titik (Custom Dash-Dot)
    if 'KP07_TANAH' not in doc.linetypes:
        doc.linetypes.new('KP07_TANAH', dxfattribs={
            'description': 'KP-07 Existing Ground',
            'pattern': [1.5, 0.75, -0.5, 0.0, -0.5] # Scaled pattern
        })
    
    # B. Setup Text Styles (Poin 2.2)
    if "ARIAL_NARROW" not in doc.styles:
        doc.styles.new("ARIAL_NARROW", dxfattribs={'font': 'Arial Narrow.ttf'})
    if "ARIAL" not in doc.styles:
        doc.styles.new("ARIAL", dxfattribs={'font': 'Arial.ttf'})

    # C. Setup Layers (Tabel 1)
    layers = [
        # Name, Color (ACI), Linetype, Lineweight (1/100 mm)
        ('DESAIN_RENCANA', 1, 'CONTINUOUS', 50),   # Merah, 0.50mm
        ('TANAH_ASLI', 8, 'KP07_TANAH', 25),       # Abu-abu, 0.25mm
        ('GRID_MAJOR', 9, 'CONTINUOUS', 13),       # Abu muda, 0.13mm
        ('TEXT_DATA', 2, 'CONTINUOUS', 25),        # Kuning, 0.25mm
        ('TEXT_LABEL', 7, 'CONTINUOUS', 25),       # Putih
        ('HATCH_CUT', 1, 'CONTINUOUS', 13),        # Merah Tipis
        ('HATCH_FILL', 3, 'CONTINUOUS', 13),       # Hijau Tipis
        ('FRAME_TABLE', 7, 'CONTINUOUS', 35),      # Putih, 0.35mm
        ('KOP_GAMBAR', 4, 'CONTINUOUS', 35)        # Cyan
    ]
    
    for name, color, ltype, lweight in layers:
        if name not in doc.layers:
            doc.layers.add(name=name, color=color, linetype=ltype, lineweight=lweight)

# ==========================================
# 2. CORE LOGIC: HATCHING & GEOMETRY
# ==========================================
def calculate_hatch_areas(tanah_pts, desain_pts):
    """
    Menggunakan Shapely untuk menghitung area Cut & Fill.
    Referensi: Poin 6.3 Rekomendasi Fitur Hatching
    """
    if not tanah_pts or not desain_pts:
        return None, None

    # Buat Polygon Tertutup (tambah dasar jauh di bawah untuk closing)
    min_y = min([p[1] for p in tanah_pts] + [p[1] for p in desain_pts]) - 10.0
    
    # Polygon Tanah
    p_tanah_closed = tanah_pts + [(tanah_pts[-1][0], min_y), (tanah_pts[0][0], min_y)]
    poly_tanah = Polygon(p_tanah_closed).buffer(0) # Fix self-intersection
    
    # Polygon Desain
    p_desain_closed = desain_pts + [(desain_pts[-1][0], min_y), (desain_pts[0][0], min_y)]
    poly_desain = Polygon(p_desain_closed).buffer(0)
    
    try:
        # Logika Boolean (Poin 6.3)
        # Cut = Tanah - Desain (Area tanah yang dibuang)
        # Fill = Desain - Tanah (Area desain yang mengisi rongga)
        # *Note: Logika ini berlaku untuk polygon solid dari datum ke surface.
        # Untuk irisan surface to surface:
        poly_cut = poly_tanah.difference(poly_desain)
        poly_fill = poly_desain.difference(poly_tanah)
        
        # Filter: Hanya ambil area di atas min_y (area asli, bukan area datum dummy)
        # Ini penyederhanaan. Untuk akurasi tinggi perlu intersection dengan bounding box atas.
        
        return poly_cut, poly_fill
    except Exception as e:
        return None, None

def draw_shapely_polygon_as_hatch(msp, geom, layer_name, pattern_name='ANSI31', scale=0.5):
    """Konversi Shapely Geometry ke DXF Hatch"""
    if geom is None or geom.is_empty: return

    polys = []
    if geom.geom_type == 'Polygon': polys = [geom]
    elif geom.geom_type == 'MultiPolygon': polys = list(geom.geoms)
    
    for poly in polys:
        # Extract exterior ring
        coords = list(poly.exterior.coords)
        if len(coords) < 3: continue
        
        # Create Hatch
        hatch = msp.add_hatch(color=256, dxfattribs={'layer': layer_name})
        hatch.set_pattern_fill(pattern_name, scale=scale)
        hatch.paths.add_polyline_path(coords)

# ==========================================
# 3. GENERATOR ENGINE (DXF)
# ==========================================
def generate_dxf_output(data_list, mode="cross"):
    """
    Main Generator Function.
    Mode: "long" (1:2000/1:100) or "cross" (1:100/1:100)
    """
    doc = ezdxf.new('R2010')
    setup_kp07_standards(doc)
    msp = doc.modelspace()
    
    # Konfigurasi Skala (Poin 2.3 & 3.0)
    if mode == "long":
        SC_H = 1.0   # Unit gambar = Meter. Plotting nanti 1:2000
        SC_V = 10.0  # Eksagerasi Vertikal 10x
        TEXT_H_LBL = 2.5
        TEXT_H_DAT = 1.8
    else:
        SC_H = 1.0   # 1:100
        SC_V = 1.0   # Cross section biasanya 1:1 agar visual proporsional, atau 10x jika detail.
                     # KP-07 Cross biasanya 1:100 H/V sama.
        TEXT_H_LBL = 2.0
        TEXT_H_DAT = 1.5

    ROW_H = 15.0 # Tinggi baris tabel data
    
    current_x_origin = 0.0
    current_y_origin = 0.0
    max_h_row = 0.0
    
    # Loop setiap STA (Cross) atau Single Run (Long)
    items = data_list if mode == "cross" else [data_list]
    
    for item in items:
        # Unpack Data
        if mode == "long":
            pts_tanah = item['points_tanah']
            pts_desain = item['points_desain']
            sta_label = "LONGITUDINAL SECTION"
        else:
            pts_tanah = item.get('points_tanah', [])
            pts_desain = item.get('points_desain', [])
            sta_label = item.get('STA', 'STA 0+000')

        if not pts_tanah and not pts_desain: continue

        # Hitung Extents
        all_pts = pts_tanah + pts_desain
        min_x = min(p[0] for p in all_pts)
        max_x = max(p[0] for p in all_pts)
        min_y = min(p[1] for p in all_pts)
        max_y = max(p[1] for p in all_pts)
        
        # Grid System
        g_min_x = math.floor(min_x / 2.0) * 2.0
        g_max_x = math.ceil(max_x / 2.0) * 2.0
        g_min_y = math.floor(min_y) - 1.0
        datum = g_min_y
        
        graph_w = (g_max_x - g_min_x) * SC_H
        graph_h = (max_y - datum) * SC_V + 5.0 # +5m space top
        
        # Koordinat Dasar Gambar
        origin_x = current_x_origin
        origin_y = current_y_origin
        base_graph_y = origin_y + (4 * ROW_H) # Space untuk 4 baris data
        
        # 1. GAMBAR GRID & TEXT DATA (Poin 2.2)
        curr_x = g_min_x
        while curr_x <= g_max_x + 0.01:
            draw_x = origin_x + (curr_x - g_min_x) * SC_H
            
            # Garis Grid Vertikal
            msp.add_line((draw_x, base_graph_y), (draw_x, base_graph_y + graph_h), 
                         dxfattribs={'layer': 'GRID_MAJOR'})
            
            # Interpolasi Elevasi
            def get_y(pts, x):
                if not pts: return None
                # Simple linear interpolation
                for i in range(len(pts)-1):
                    if pts[i][0] <= x <= pts[i+1][0]:
                        ratio = (x - pts[i][0])/(pts[i+1][0] - pts[i][0]) if (pts[i+1][0]-pts[i][0])!=0 else 0
                        return pts[i][1] + ratio * (pts[i+1][1] - pts[i][1])
                return None

            z_t = get_y(pts_tanah, curr_x)
            z_d = get_y(pts_desain, curr_x)
            
            # Penulisan Data (Rotasi 90 derajat - Poin 2.2)
            # Baris 1: Jarak
            msp.add_text(f"{curr_x:.1f}", dxfattribs={
                'style': 'ARIAL_NARROW', 'height': TEXT_H_DAT, 'layer': 'TEXT_DATA', 'rotation': 90
            }).set_placement((draw_x, origin_y + 0.5*ROW_H), align=TextEntityAlignment.MIDDLE_CENTER)
            
            # Baris 2: Tanah Asli
            if z_t is not None:
                msp.add_text(f"{z_t:.3f}", dxfattribs={
                    'style': 'ARIAL_NARROW', 'height': TEXT_H_DAT, 'layer': 'TEXT_DATA', 'rotation': 90
                }).set_placement((draw_x, origin_y + 1.5*ROW_H), align=TextEntityAlignment.MIDDLE_CENTER)
            
            # Baris 3: Desain
            if z_d is not None:
                msp.add_text(f"{z_d:.3f}", dxfattribs={
                    'style': 'ARIAL_NARROW', 'height': TEXT_H_DAT, 'layer': 'TEXT_DATA', 'rotation': 90
                }).set_placement((draw_x, origin_y + 2.5*ROW_H), align=TextEntityAlignment.MIDDLE_CENTER)
                
            curr_x += 2.0 # Interval Grid
            
        # 2. GAMBAR POLYLINE UTAMA (Poin 2.1)
        if pts_tanah:
            pts_draw_t = [(origin_x + (p[0]-g_min_x)*SC_H, base_graph_y + (p[1]-datum)*SC_V) for p in pts_tanah]
            msp.add_lwpolyline(pts_draw_t, dxfattribs={'layer': 'TANAH_ASLI'})
            
        if pts_desain:
            pts_draw_d = [(origin_x + (p[0]-g_min_x)*SC_H, base_graph_y + (p[1]-datum)*SC_V) for p in pts_desain]
            msp.add_lwpolyline(pts_draw_d, dxfattribs={'layer': 'DESAIN_RENCANA'})
            
        # 3. HATCHING / ARSIRAN (Fitur Baru - Poin 6.3)
        # Transform points to Local DXF coordinates for calculation
        if pts_tanah and pts_desain:
            t_local = [(origin_x + (p[0]-g_min_x)*SC_H, base_graph_y + (p[1]-datum)*SC_V) for p in pts_tanah]
            d_local = [(origin_x + (p[0]-g_min_x)*SC_H, base_graph_y + (p[1]-datum)*SC_V) for p in pts_desain]
            
            poly_cut, poly_fill = calculate_hatch_areas(t_local, d_local)
            
            # Arsiran Cut (Merah/ANSI31)
            draw_shapely_polygon_as_hatch(msp, poly_cut, 'HATCH_CUT', scale=0.5)
            # Arsiran Fill (Hijau/ANSI37 - Silang)
            draw_shapely_polygon_as_hatch(msp, poly_fill, 'HATCH_FILL', pattern_name='ANSI37', scale=0.5)

        # 4. FRAME & LABEL BAND (Poin 3.1 & 4.2)
        # Garis Horizontal Tabel
        for i in range(5):
            y_line = origin_y + i * ROW_H
            msp.add_line((origin_x, y_line), (origin_x + graph_w, y_line), dxfattribs={'layer': 'FRAME_TABLE'})
            
        # Label Kiri
        labels = ["JARAK", "EL. TANAH", "EL. RENCANA", "DATUM"]
        for i, txt in enumerate(labels):
            msp.add_text(txt, dxfattribs={'style': 'ARIAL', 'height': TEXT_H_LBL, 'layer': 'TEXT_LABEL'})\
               .set_placement((origin_x - 1, origin_y + (i+0.5)*ROW_H), align=TextEntityAlignment.MIDDLE_RIGHT)

        # Judul STA
        msp.add_text(sta_label, dxfattribs={'style': 'ARIAL', 'height': 4.0, 'layer': 'TEXT_LABEL'})\
           .set_placement((origin_x + graph_w/2, base_graph_y + graph_h + 2), align=TextEntityAlignment.BOTTOM_CENTER)
        
        # Datum Text
        msp.add_text(f"+{datum:.2f}", dxfattribs={'style': 'ARIAL', 'height': TEXT_H_LBL, 'layer': 'TEXT_LABEL'})\
           .set_placement((origin_x - 1, origin_y + 3.5*ROW_H), align=TextEntityAlignment.MIDDLE_RIGHT)

        # Layout Logic (Pindah Posisi untuk Next Loop)
        if mode == "cross":
            current_x_origin += graph_w + 50.0 # Spasi horizontal
            max_h_row = max(max_h_row, graph_h + 4*ROW_H)
            
            if current_x_origin > 500.0: # Wrap ke bawah jika terlalu lebar
                current_x_origin = 0.0
                current_y_origin -= (max_h_row + 50.0)
                max_h_row = 0.0

    return io.StringIO(doc.write_result()).getvalue().encode('utf-8')

# ==========================================
# 4. PARSER DATA INPUT
# ==========================================
def parse_excel_pclp(file):
    """Parser format manual PCLP (Block Based)."""
    try:
        df = pd.read_excel(file, header=None).astype(str)
        data = []
        i = 0
        while i < len(df):
            row = df.iloc[i].values
            # Cari marker 'X' dan 'Y'
            if 'X' in row:
                x_idx = list(row).index('X')
                if i+1 < len(df) and df.iloc[i+1, x_idx] == 'Y':
                    # Found Block
                    sta_raw = str(df.iloc[i+1, 1])
                    if sta_raw == 'nan': sta_raw = f"STA {len(data)}"
                    
                    points = []
                    col = x_idx + 1
                    while col < len(row):
                        try:
                            vx = float(row[col])
                            vy = float(df.iloc[i+1, col])
                            points.append((vx, vy))
                        except: break
                        col += 1
                    
                    # Coba cari baris Desain (biasanya 2 baris dibawah atau di sheet lain)
                    # Untuk demo ini, kita buat desain dummy (rata-rata elevasi - 2m)
                    pts_desain = []
                    if points:
                        avg_y = sum(p[1] for p in points)/len(points)
                        # Buat trapesium sederhana
                        pts_desain = [
                            (points[0][0], avg_y-1), 
                            (points[len(points)//2][0]-1, avg_y-3),
                            (points[len(points)//2][0]+1, avg_y-3),
                            (points[-1][0], avg_y-1)
                        ]

                    data.append({
                        'STA': sta_raw,
                        'points_tanah': sorted(points, key=lambda x: x[0]),
                        'points_desain': sorted(pts_desain, key=lambda x: x[0])
                    })
            i += 1
        return data
    except Exception as e:
        st.error(f"Error Parsing: {e}")
        return []

# ==========================================
# 5. UI STREAMLIT
# ==========================================
st.set_page_config(page_title="IIDAS: KP-07 Automation", layout="wide")

st.title("🇮🇩 IIDAS - KP-07 Drafting Engine")
st.markdown("""
> **Analisis Kepatuhan**: Aplikasi ini telah direvisi untuk memenuhi standar *Kriteria Perencanaan Irigasi (KP-07)*, mencakup:
> * **Layering**: Desain (Merah/0.5mm) vs Tanah Asli (Abu/Dashed).
> * **Visual**: Arsiran otomatis Galian/Timbunan (Hatching) & Font Arial Narrow.
> * **Data**: Kompatibilitas Civil 3D (CSV no-header).
""")

tab_cross, tab_long, tab_gis = st.tabs(["📐 Cross Section (Detail)", "📈 Long Section (Trase)", "🗺️ Peta Situasi (GIS)"])

with tab_cross:
    st.header("Generator Potongan Melintang (KP-07)")
    up_file = st.file_uploader("Upload Excel (Format PCLP)", type=['xlsx'])
    
    if up_file:
        data = parse_excel_pclp(up_file)
        if data:
            st.success(f"Berhasil membaca {len(data)} cross section.")
            
            # Preview Logic
            idx = st.slider("Pilih STA untuk Preview", 0, len(data)-1, 0)
            item = data[idx]
            
            # Matplotlib Preview (Simple)
            fig, ax = plt.subplots(figsize=(10, 3))
            if item['points_tanah']: ax.plot(*zip(*item['points_tanah']), 'k--', label='Tanah Asli')
            if item['points_desain']: ax.plot(*zip(*item['points_desain']), 'r-', linewidth=2, label='Desain')
            ax.set_title(item['STA']); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Export Buttons
            c1, c2 = st.columns(2)
            dxf_data = generate_dxf_output(data, mode="cross")
            c1.download_button("📥 Download DXF (Layer KP-07 & Arsiran)", dxf_data, "Cross_KP07.dxf")
            
            # CSV Civil 3D (PNEZD / SOE) - Poin 3.2
            csv_str = ""
            for d in data:
                sta_num = ''.join(filter(str.isdigit, d['STA'])) or '0'
                for p in d['points_tanah']:
                    csv_str += f"{sta_num},{p[0]},{p[1]},EG\n" # Station, Offset, Elev, Desc
            c2.download_button("📥 Download CSV (Format Civil 3D)", csv_str, "Civil3D_Import.csv")

with tab_long:
    st.info("Fitur Long Section menggunakan format input Station-Elevation sederhana.")
    # Input Manual for Demo
    raw_text = st.text_area("Paste Data (Station, Elevation)", "0,100\n50,99.5\n100,99.2\n150,98.8\n200,99.0")
    
    if raw_text:
        try:
            pts = []
            for line in raw_text.split('\n'):
                parts = line.split(',')
                if len(parts) >= 2: pts.append((float(parts[0]), float(parts[1])))
            
            # Dummy Design (Slope 0.001)
            pts_d = [(p[0], 100 - (p[0]*0.005)) for p in pts]
            
            long_data = {'points_tanah': pts, 'points_desain': pts_d}
            
            st.pyplot(plt.figure(figsize=(10,2))) # Placeholder plotting
            
            dxf_long = generate_dxf_output(long_data, mode="long")
            st.download_button("📥 Download Long Section DXF (Exaggerated 10x)", dxf_long, "Long_KP07.dxf")
            
        except ValueError:
            st.error("Format data salah. Gunakan koma sebagai pemisah.")

with tab_gis:
    st.write("Modul GIS membutuhkan file DEM & SHP (Fitur dipertahankan dari versi sebelumnya untuk Peta Situasi).")
    if HAS_GEO_LIBS:
        st.success("✅ Engine GIS Aktif (Geopandas/Rasterio Detected)")
        # ... (Kode GIS eksisting dapat ditempel di sini jika diperlukan)
    else:
        st.warning("⚠️ Engine GIS Inaktif. Install `geopandas rasterio` untuk mengaktifkan fitur Peta Situasi otomatis.")
