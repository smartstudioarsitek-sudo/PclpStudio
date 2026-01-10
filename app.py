import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point

# --- HANDLING LIBRARY ---
# Pastikan library ini terinstall di environment:
# pip install ezdxf geopandas rasterio shapely scipy matplotlib pandas streamlit
try:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
except ImportError:
    st.error("⚠️ Library 'ezdxf' missing. Install dengan `pip install ezdxf`")
    st.stop()

HAS_GEO_LIBS = False
try:
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes
    from scipy.ndimage import gaussian_filter
    HAS_GEO_LIBS = True
except ImportError:
    pass

# ==========================================
# 1. KONFIGURASI STANDAR KP-07 (LAYERING & STYLE)
# ==========================================
def setup_kp07_standards(doc):
    """
    Mengatur Layer, Linetype, dan Text Style sesuai mandat KP-07 & IIDAS.
    [Referensi Spesifikasi: Tabel 1 & Poin 2.1]
    """
    # A. Setup Linetypes
    if 'KP07_TANAH' not in doc.linetypes:
        doc.linetypes.new('KP07_TANAH', dxfattribs={
            'description': 'Existing Ground (Chain Line)',
            'pattern': [1.0, -0.5, 0.0, -0.5] # Garis-Spasi-Titik-Spasi
        })
    if 'CENTER' not in doc.linetypes:
        doc.linetypes.new('CENTER', dxfattribs={'description': 'Center', 'pattern': [1.25, -0.25, 0.25, -0.25]})
    if 'PHANTOM' not in doc.linetypes:
        doc.linetypes.new('PHANTOM', dxfattribs={'description': 'Phantom', 'pattern': [1.25, -0.25, 0.25, -0.25, 0.25, -0.25]})

    # B. Setup Text Styles (Arial Narrow untuk Data Padat)
    if "ARIAL_NARROW" not in doc.styles:
        doc.styles.new("ARIAL_NARROW", dxfattribs={'font': 'Arial Narrow.ttf'})
    if "ARIAL" not in doc.styles:
        doc.styles.new("ARIAL", dxfattribs={'font': 'Arial.ttf'})

    # C. Setup Layers
    layers = [
        # Nama Layer, Warna (ACI), Tipe Garis, Tebal (1/100mm)
        ('DESAIN_RENCANA', 1, 'CONTINUOUS', 50),    # Merah, Tebal (Desain)
        ('TANAH_ASLI', 8, 'KP07_TANAH', 25),        # Abu, Tipis (Eksisting)
        ('GRID_MAJOR', 9, 'CONTINUOUS', 13),        # Abu Muda (Grid)
        ('TEXT_DATA', 2, 'CONTINUOUS', 25),         # Kuning (Angka Tabel)
        ('TEXT_LABEL', 7, 'CONTINUOUS', 25),        # Putih (Label/Judul)
        ('HATCH_CUT', 1, 'CONTINUOUS', 13),         # Merah (Arsir Galian)
        ('HATCH_FILL', 3, 'CONTINUOUS', 13),        # Hijau (Arsir Timbunan)
        ('FRAME_TABLE', 7, 'CONTINUOUS', 35),       # Frame Tabel
        ('KOP_GAMBAR', 7, 'CONTINUOUS', 35),        # Kop Gambar
        ('SITUASI_AS', 1, 'CENTER', 35),            # Peta Situasi: As
        ('SITUASI_POT', 6, 'PHANTOM', 25),          # Peta Situasi: Garis Potong
        ('SITUASI_KONTUR_MJR', 3, 'CONTINUOUS', 25),# Peta Situasi: Kontur Mayor
        ('SITUASI_KONTUR_MNR', 9, 'CONTINUOUS', 13) # Peta Situasi: Kontur Minor
    ]
    
    for name, color, ltype, lweight in layers:
        if name not in doc.layers:
            doc.layers.add(name=name, color=color, linetype=ltype, lineweight=lweight)

# ==========================================
# 2. ENGINE GEOSPATIAL (GIS TO CAD)
# ==========================================
def extract_gis_data(dem_file, shp_file, interval=50, w_left=25, w_right=25):
    """
    Ekstraksi Long/Cross Section & Generasi Garis Potong dari DEM + SHP.
    [Referensi Spesifikasi: Poin 5.1 & 4.1]
    """
    if not HAS_GEO_LIBS: return None, None, None, "Modul GIS tidak terinstall."
    
    try:
        with rasterio.open(dem_file) as src:
            # Baca DEM & Smoothing (Poin 4.1)
            dem_arr = src.read(1)
            transform = src.transform
            nodata = src.nodata
            
            # Smoothing ringan untuk menghilangkan noise permukaan mikro
            dem_smooth = gaussian_filter(dem_arr, sigma=1) 
            
            gdf = gpd.read_file(shp_file)
            if gdf.crs != src.crs: gdf = gdf.to_crs(src.crs)
            
            # Ambil Geometri LineString Pertama
            line_geom = gdf.geometry.iloc[0]
            if line_geom.geom_type == 'MultiLineString': line_geom = line_geom.geoms[0]
            
            length = line_geom.length
            stations = np.arange(0, length, interval)
            
            long_data = []
            cross_data_list = []
            cut_lines_dxf = []
            
            for dist in stations:
                # 1. Interpolasi Titik Center
                pt_center = line_geom.interpolate(dist)
                
                # Sampling Elevasi Center (Untuk Long Section)
                row, col = src.index(pt_center.x, pt_center.y)
                z_center = dem_smooth[row, col] if (0 <= row < src.height and 0 <= col < src.width) else np.nan
                if z_center == nodata: z_center = np.nan
                
                long_data.append((dist, z_center))
                
                # 2. Hitung Vektor Tegak Lurus (Poin 5.1)
                # Ambil titik +/- 0.1m untuk cari tangen
                p_back = line_geom.interpolate(max(0, dist - 0.5))
                p_front = line_geom.interpolate(min(length, dist + 0.5))
                dx, dy = p_front.x - p_back.x, p_front.y - p_back.y
                length_vec = math.sqrt(dx**2 + dy**2)
                
                if length_vec > 0:
                    # Normal Vector (-dy, dx)
                    nx, ny = -dy/length_vec, dx/length_vec
                    
                    # Buat Garis Potong (Cut Line) untuk Peta Situasi
                    p_left_global = (pt_center.x + nx * -w_left, pt_center.y + ny * -w_left)
                    p_right_global = (pt_center.x + nx * w_right, pt_center.y + ny * w_right)
                    
                    cut_lines_dxf.append({
                        'sta': f"STA {int(dist)}",
                        'geometry': [p_left_global, p_right_global]
                    })
                    
                    # 3. Sampling Cross Section
                    # Loop offset dari kiri (-) ke kanan (+)
                    offsets = np.arange(-w_left, w_right+1, 1.0) # Step 1 meter
                    pts_cross = []
                    
                    for off in offsets:
                        sx = pt_center.x + nx * off
                        sy = pt_center.y + ny * off
                        r, c = src.index(sx, sy)
                        val = dem_smooth[r, c] if (0 <= r < src.height and 0 <= c < src.width) else np.nan
                        if val == nodata: val = np.nan
                        
                        if not np.isnan(val):
                            pts_cross.append((off, val)) # Format: (Offset, Elevation)
                    
                    # Simpan Data Cross
                    # Mockup Desain: Kanal trapesium sederhana (misal dalam 2m)
                    z_des = z_center - 2.0 if not np.isnan(z_center) else np.nan
                    pts_desain = []
                    if not np.isnan(z_des):
                         # B=1.0, m=1.0, H=2.0
                         pts_desain = [(-5, z_center), (-3, z_center), (-1.5, z_des), (1.5, z_des), (3, z_center), (5, z_center)]

                    cross_data_list.append({
                        'STA': f"STA {int(dist)}+00",
                        'points_tanah': pts_cross,
                        'points_desain': pts_desain # Placeholder desain otomatis
                    })

            return long_data, cross_data_list, (gdf, cut_lines_dxf), None

    except Exception as e:
        return None, None, None, str(e)

def generate_contours_dxf(msp, dem_file, interval=1.0):
    """
    Generate Kontur dari DEM dan inject ke DXF Modelspace.
    [Referensi Spesifikasi: Poin 4.2]
    """
    if not HAS_GEO_LIBS: return
    try:
        with rasterio.open(dem_file) as src:
            arr = src.read(1)
            # Smoothing wajib untuk kontur yang bagus (Poin 4.1)
            arr = gaussian_filter(arr, sigma=1.0)
            
            # Gunakan Matplotlib untuk generate path kontur
            min_val, max_val = np.nanmin(arr), np.nanmax(arr)
            levels = np.arange(math.floor(min_val), math.ceil(max_val), interval)
            
            # Kita pakai plt.contour di backend (tanpa plot ke layar) untuk dapat pathnya
            fig_dummy = plt.figure()
            ax_dummy = fig_dummy.add_subplot(111)
            cs = ax_dummy.contour(arr, levels=levels, extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top))
            
            for level, collection in zip(levels, cs.collections):
                # Tentukan Layer (Mayor/Minor)
                is_major = (int(level) % 5 == 0)
                layer_name = 'SITUASI_KONTUR_MJR' if is_major else 'SITUASI_KONTUR_MNR'
                
                for path in collection.get_paths():
                    if len(path.vertices) < 2: continue
                    # Konversi path ke LWPolyline DXF dengan elevasi
                    # [Poin 4.2: DXF Polyline with Elevation Attribute]
                    msp.add_lwpolyline(path.vertices, dxfattribs={
                        'layer': layer_name,
                        'elevation': float(level) 
                    })
            plt.close(fig_dummy)
    except Exception as e:
        st.error(f"Gagal generate kontur: {e}")

# ==========================================
# 3. ENGINE DRAFTING & HATCHING (CORE)
# ==========================================
def calculate_hatch(tanah, desain):
    """Logika Boolean Shapely untuk Arsiran Cut/Fill [Poin 6.3]"""
    if not tanah or not desain: return None, None
    min_y = min([p[1] for p in tanah] + [p[1] for p in desain]) - 10
    
    # Polygon Tertutup ke Datum
    p_t = tanah + [(tanah[-1][0], min_y), (tanah[0][0], min_y)]
    p_d = desain + [(desain[-1][0], min_y), (desain[0][0], min_y)]
    
    poly_t = Polygon(p_t).buffer(0)
    poly_d = Polygon(p_d).buffer(0)
    
    try:
        cut = poly_t.difference(poly_d) # Tanah dibuang
        fill = poly_d.difference(poly_t) # Desain mengisi rongga
        return cut, fill
    except: return None, None

def draw_shapely_hatch(msp, geom, layer, pattern='ANSI31', scale=0.5, angle=0):
    if geom is None or geom.is_empty: return
    polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    for p in polys:
        if p.area < 0.01: continue
        hatch = msp.add_hatch(color=256, dxfattribs={'layer': layer})
        hatch.set_pattern_fill(pattern, scale=scale, angle=angle)
        hatch.paths.add_polyline_path(list(p.exterior.coords))

def generate_profile_dxf_final(data, mode="cross"):
    """Generator DXF Profil (Cross/Long) dengan Arsiran & Standar KP-07"""
    doc = ezdxf.new('R2010')
    setup_kp07_standards(doc)
    msp = doc.modelspace()
    
    SC_H = 1.0
    SC_V = 10.0 if mode == "long" else 1.0 # Vert. Exaggeration 10x untuk Long
    ROW_H = 15.0
    
    curr_x, curr_y = 0, 0
    max_h_row = 0
    
    items = data if mode == "cross" else [data] # Data wrapper
    
    for item in items:
        pts_t = item.get('points_tanah', [])
        pts_d = item.get('points_desain', [])
        sta = item.get('STA', 'STA')
        
        if not pts_t and not pts_d: continue
        
        # Bounds & Grid
        all_p = pts_t + pts_d
        min_x, max_x = min(p[0] for p in all_p), max(p[0] for p in all_p)
        min_y, max_y = min(p[1] for p in all_p), max(p[1] for p in all_p)
        
        g_min_x = math.floor(min_x/2)*2
        g_max_x = math.ceil(max_x/2)*2
        g_min_y = math.floor(min_y)-1
        datum = g_min_y
        
        w_draw = (g_max_x - g_min_x) * SC_H
        h_draw = (max_y - datum) * SC_V + 5
        
        ox, oy = curr_x, curr_y
        base_y = oy + 4*ROW_H
        
        # 1. Grid & Data
        gx = g_min_x
        while gx <= g_max_x + 0.01:
            dx = ox + (gx - g_min_x)*SC_H
            msp.add_line((dx, base_y), (dx, base_y+h_draw), dxfattribs={'layer': 'GRID_MAJOR'})
            
            # Interpolasi
            def get_z(pts, x):
                for i in range(len(pts)-1):
                    if pts[i][0] <= x <= pts[i+1][0]:
                        ratio = (x - pts[i][0])/(pts[i+1][0] - pts[i][0]) if (pts[i+1][0]-pts[i][0]) !=0 else 0
                        return pts[i][1] + ratio * (pts[i+1][1] - pts[i][1])
                return None
            
            zt = get_z(pts_t, gx)
            zd = get_z(pts_d, gx)
            
            # Text Style (Rotated)
            sty = {'style': 'ARIAL_NARROW', 'height': 1.8, 'layer': 'TEXT_DATA', 'rotation': 90}
            msp.add_text(f"{gx:.1f}", dxfattribs=sty).set_placement((dx, oy+0.5*ROW_H), align=TextEntityAlignment.MIDDLE_CENTER)
            if zt: msp.add_text(f"{zt:.2f}", dxfattribs=sty).set_placement((dx, oy+1.5*ROW_H), align=TextEntityAlignment.MIDDLE_CENTER)
            if zd: msp.add_text(f"{zd:.2f}", dxfattribs=sty).set_placement((dx, oy+2.5*ROW_H), align=TextEntityAlignment.MIDDLE_CENTER)
            
            gx += 2.0
            
        # 2. Geometry & Hatch
        t_loc = [(ox+(p[0]-g_min_x)*SC_H, base_y+(p[1]-datum)*SC_V) for p in pts_t]
        d_loc = [(ox+(p[0]-g_min_x)*SC_H, base_y+(p[1]-datum)*SC_V) for p in pts_d]
        
        cut, fill = calculate_hatch(t_loc, d_loc)
        draw_shapely_hatch(msp, cut, 'HATCH_CUT', 'ANSI31')
        draw_shapely_hatch(msp, fill, 'HATCH_FILL', 'ANSI37', angle=45)
        
        if t_loc: msp.add_lwpolyline(t_loc, dxfattribs={'layer': 'TANAH_ASLI'})
        if d_loc: msp.add_lwpolyline(d_loc, dxfattribs={'layer': 'DESAIN_RENCANA'})
        
        # 3. Frame & Labels
        for i in range(5):
            msp.add_line((ox, oy+i*ROW_H), (ox+w_draw, oy+i*ROW_H), dxfattribs={'layer': 'FRAME_TABLE'})
            
        lbls = ["JARAK", "EL. TANAH", "EL. DESAIN", "DATUM"]
        for i, l in enumerate(lbls):
            msp.add_text(l, dxfattribs={'style': 'ARIAL', 'height': 2.0, 'layer': 'TEXT_LABEL'}).set_placement((ox-1, oy+(i+0.5)*ROW_H), align=TextEntityAlignment.MIDDLE_RIGHT)
            
        msp.add_text(sta, dxfattribs={'style': 'ARIAL', 'height': 3.5, 'layer': 'TEXT_LABEL'}).set_placement((ox+w_draw/2, base_y+h_draw+2), align=TextEntityAlignment.BOTTOM_CENTER)
        msp.add_text(f"+{datum:.2f}", dxfattribs={'style': 'ARIAL', 'height': 2.0, 'layer': 'TEXT_LABEL'}).set_placement((ox-1, oy+3.5*ROW_H), align=TextEntityAlignment.MIDDLE_RIGHT)
        
        # Layout Iteration
        if mode == "cross":
            curr_x += w_draw + 50
            max_h_row = max(max_h_row, h_draw + 4*ROW_H)
            if curr_x > 500:
                curr_x = 0
                curr_y -= (max_h_row + 50)
                max_h_row = 0

    return io.StringIO(doc.write_result()).getvalue().encode('utf-8')

def generate_situasi_final_dxf(gdf_trase, cut_lines, dem_file=None):
    """
    Generator Peta Situasi Lengkap (Trase + Garis Potong + Kontur)
    [Referensi Spesifikasi: Poin 5.2]
    """
    doc = ezdxf.new('R2010')
    setup_kp07_standards(doc)
    msp = doc.modelspace()
    
    # 1. Gambar Kontur (Jika DEM ada)
    if dem_file:
        generate_contours_dxf(msp, dem_file, interval=1.0)
    
    # 2. Gambar Trase (As Saluran)
    if not gdf_trase.empty:
        geom = gdf_trase.geometry.iloc[0]
        if geom.geom_type == 'MultiLineString': geom = geom.geoms[0]
        msp.add_lwpolyline(list(geom.coords), dxfattribs={'layer': 'SITUASI_AS'})
        
    # 3. Gambar Garis Potongan (Cut Lines) - Poin 5.1
    if cut_lines:
        for cl in cut_lines:
            pts = cl['geometry']
            msp.add_line(pts[0], pts[1], dxfattribs={'layer': 'SITUASI_POT'})
            
            # Label STA (Rotated)
            ang = math.degrees(math.atan2(pts[1][1]-pts[0][1], pts[1][0]-pts[0][0]))
            if not (-90 < ang <= 90): ang += 180 # Text readability
            msp.add_text(cl['sta'], dxfattribs={
                'height': 2.0, 'layer': 'TEXT_LABEL', 'rotation': ang
            }).set_placement(pts[1], align=TextEntityAlignment.BOTTOM_LEFT)
            
    return io.StringIO(doc.write_result()).getvalue().encode('utf-8')

# ==========================================
# 4. INTEROPERABILITAS CIVIL 3D (CSV EXPORT)
# ==========================================
def generate_civil3d_csv(data_list):
    """
    Export CSV khusus untuk Import Civil 3D (Station-Offset-Elevation).
    [Referensi Spesifikasi: Poin 3.2 - Opsi B]
    Penting: Offset Kiri harus Negatif.
    """
    output = io.StringIO()
    # Format: Station, Offset, Elevation, Description
    # Tidak boleh ada Header untuk import Raw Points tertentu, tapi PNEZD biasanya butuh format standar.
    # Kita pakai format: Station,Offset,Elevation,Description (SOE Format)
    
    for item in data_list:
        try:
            # Bersihkan String STA (misal "STA 0+100" -> 100.0)
            sta_str = item['STA'].upper().replace('STA', '').replace(' ', '').replace('+', '')
            if len(sta_str) > 3: # Asumsi format 0100 -> 100
                sta_val = float(sta_str[:-3] + sta_str[-3:]) # Handle simple parsing
            else:
                sta_val = float(sta_str)
        except: sta_val = 0.0

        pts = item.get('points_tanah', [])
        for offset, elev in pts:
            # offset sudah bertanda (+/-) dari proses GIS
            desc = "EG" # Existing Ground
            output.write(f"{sta_val},{offset},{elev},{desc}\n")
            
        pts_d = item.get('points_desain', [])
        for offset, elev in pts_d:
            desc = "FG" # Finish Ground
            output.write(f"{sta_val},{offset},{elev},{desc}\n")
            
    return output.getvalue().encode('utf-8')

# ==========================================
# 5. UI UTAMA (STREAMLIT)
# ==========================================
st.set_page_config(page_title="IIDAS Master v8", layout="wide")
st.title("🛡️ IIDAS Master: KP-07 & Civil 3D Integrator")
st.markdown("""
**Status Sistem:**
* ✅ **GIS Engine**: DEM Processing, Contour Generation, Cut Lines Orthogonal [Specs 4.1, 5.1]
* ✅ **Drafting Engine**: KP-07 Layers, Auto Hatching (Cut/Fill), Typography [Specs 2.1, 6.3]
* ✅ **Interoperability**: Civil 3D CSV Export (SOE Format) [Specs 3.2]
""")

if not HAS_GEO_LIBS:
    st.warning("⚠️ **Mode Terbatas**: Library GIS (geopandas/rasterio) tidak terdeteksi. Fitur Peta Situasi & Auto-Extract dinonaktifkan. Silakan upload data Excel manual.")

tabs = st.tabs(["🗺️ GIS & PETA SITUASI", "📐 CROSS SECTION", "📈 LONG SECTION"])

# --- TAB 1: GIS ---
with tabs[0]:
    st.header("Analisis Spasial & Situasi")
    c1, c2 = st.columns([1,2])
    with c1:
        up_dem = st.file_uploader("1. Upload DEM (.tif)", type=['tif'])
        up_shp = st.file_uploader("2. Upload Trase (.shp/.geojson)", type=['shp', 'geojson', 'zip'])
        
        st.write("---")
        interval = st.number_input("Interval Cross (m)", 10, 1000, 50)
        w_swath = st.number_input("Lebar Pemeriksaan (m)", 5, 200, 25)
        
        btn_gis = st.button("🚀 PROSES GIS")
        
    with c2:
        if btn_gis and up_dem and up_shp and HAS_GEO_LIBS:
            with st.spinner("Processing DEM, Smoothing Terrain, Calculating Vectors..."):
                # Save temp for rasterio
                with open("temp_dem.tif", "wb") as f: f.write(up_dem.getbuffer())
                with open("temp_trase.geojson", "wb") as f: f.write(up_shp.getbuffer())
                
                long_res, cross_res, situ_res, err = extract_gis_data("temp_dem.tif", "temp_trase.geojson", interval, w_swath, w_swath)
                
                if err:
                    st.error(f"GIS Error: {err}")
                else:
                    st.success(f"Analisis Selesai! Ditemukan {len(cross_res)} potongan melintang.")
                    st.session_state['data_cross'] = cross_res
                    st.session_state['data_long'] = long_res
                    st.session_state['situ_data'] = situ_res # (gdf, cut_lines)
                    
                    # Generate DXF Situasi
                    gdf_trase, cut_lines = situ_res
                    dxf_situ = generate_situasi_final_dxf(gdf_trase, cut_lines, "temp_dem.tif")
                    st.download_button("📥 DOWNLOAD DXF PETA SITUASI (Kontur + Potongan)", dxf_situ, "Peta_Situasi_Lengkap.dxf")

# --- TAB 2: CROSS SECTION ---
with tabs[1]:
    st.subheader("Output Cross Section (KP-07)")
    if 'data_cross' in st.session_state:
        data = st.session_state['data_cross']
        
        # Preview
        idx = st.slider("Preview Station Index", 0, len(data)-1, 0)
        item = data[idx]
        fig, ax = plt.subplots(figsize=(10,3))
        if item['points_tanah']: ax.plot(*zip(*item['points_tanah']), 'k--', label='Tanah')
        if item['points_desain']: ax.plot(*zip(*item['points_desain']), 'r-', label='Desain')
        ax.set_title(item['STA']); ax.grid(True); ax.legend()
        st.pyplot(fig)
        
        c1, c2 = st.columns(2)
        # DXF Output
        dxf_cross = generate_profile_dxf_final(data, "cross")
        c1.download_button("📥 DOWNLOAD DXF CROSS (KP-07)", dxf_cross, "Cross_KP07.dxf")
        
        # Civil 3D CSV
        csv_c3d = generate_civil3d_csv(data)
        c2.download_button("📥 DOWNLOAD CSV CIVIL 3D", csv_c3d, "Import_Civil3D.csv")
    else:
        st.info("Belum ada data. Silakan proses di Tab GIS atau Upload Excel Manual.")
        # Opsional: Fitur Upload Manual PCLP (bisa dicopy dari kode sebelumnya jika butuh)

# --- TAB 3: LONG SECTION ---
with tabs[2]:
    st.subheader("Output Long Section")
    if 'data_long' in st.session_state:
        # data_long format: [(dist, elev), ...]
        pts = st.session_state['data_long']
        # Bungkus ke format standar
        data_wrapper = {'points_tanah': pts, 'points_desain': [], 'STA': 'LONG SECTION'}
        
        fig, ax = plt.subplots(figsize=(12,3))
        ax.plot(*zip(*pts), 'k-')
        ax.set_title("Longitudinal Profile"); ax.grid(True)
        st.pyplot(fig)
        
        dxf_long = generate_profile_dxf_final(data_wrapper, "long")
        st.download_button("📥 DOWNLOAD DXF LONG (Scale H=1:1000 V=1:100)", dxf_long, "Long_Profile.dxf")
    else:
        st.info("Proses data GIS terlebih dahulu.")
