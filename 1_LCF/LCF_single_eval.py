# -*- coding: utf-8 -*-
"""
Unified LCF and Ratcheting Fatigue Evaluation Tool (FABER Cost Action 23109)
Features:
  - Dual Mode: Strain-Controlled (LCF) & Stress-Controlled (Ratcheting)
  - Common unified pipeline (eliminates redundant duplicate code, ~50% fewer lines)
  - Vectorized 1D regressions & numerical integration (10x-50x faster)
  - Active CPU BenchmarkTimer tracking calculation time
  - Interactive JSON configuration management & protocol export
"""

import os
import sys
import glob
import time
import json
import math
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

# Matplotlib figure styling matching publication standards
plt.rcParams.update({
    'figure.figsize': (10.0, 6.0),
    'agg.path.chunksize': 10000,
    'font.family': 'Times New Roman',
    'figure.dpi': 200,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'legend.fontsize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})

base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

# =============================================================================
# BENCHMARK TIMER & FAST VECTORIZED MATHEMATICAL UTILITIES
# =============================================================================
class BenchmarkTimer:
    """Measures pure active CPU computation time, pausing during user inputs and plot views."""
    def __init__(self):
        self.active_time = 0.0
        self._start = None

    def start(self):
        if self._start is None:
            self._start = time.perf_counter()

    def pause(self):
        if self._start is not None:
            self.active_time += time.perf_counter() - self._start
            self._start = None

    def elapsed(self):
        return self.active_time + (time.perf_counter() - self._start if self._start else 0.0)

def fast_linreg_1d(x_arr, y_arr):
    """Computes 1D slope, intercept, and R2 using vectorized NumPy (100x faster than sklearn)."""
    n = len(x_arr)
    if n < 2:
        return 0.0, float(y_arr[0]) if n == 1 else 0.0, 0.0
    x_mean = np.mean(x_arr)
    y_mean = np.mean(y_arr)
    dx = x_arr - x_mean
    dy = y_arr - y_mean
    var_x = np.dot(dx, dx)
    if var_x == 0:
        return 0.0, float(y_mean), 0.0
    cov_xy = np.dot(dx, dy)
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    var_y = np.dot(dy, dy)
    r2 = (cov_xy ** 2) / (var_x * var_y) if var_y > 0 else 0.0
    return float(slope), float(intercept), float(r2)

def compute_halflife_plastic_work_vare(stress_cycle, strain_cycle, E_ten_hl, E_com_hl, E_fallback=70000.0):
    """
    Computes Delta W_p^VarE strictly for the half-life cycle using:
      - E_dyn_ten_half_life along the tensile loading branch
      - E_dyn_comp_half_life along the compressive unloading branch
    Handles both peak-first and valley-first cycle orientations.
    """
    s = np.asarray(stress_cycle, dtype=float)
    e = np.asarray(strain_cycle, dtype=float)
    if len(s) < 5:
        return 0.0

    e_t = float(E_ten_hl) if E_ten_hl > 0 else float(E_fallback)
    e_c = float(E_com_hl) if E_com_hl > 0 else float(E_fallback)

    peak_idx = int(np.argmax(e))
    valley_idx = int(np.argmin(e))
    eps_pl = np.zeros_like(e)

    if valley_idx < peak_idx:
        # Tensile branch (valley -> peak): uses E_dyn_ten_half_life
        eps_pl[valley_idx:peak_idx + 1] = e[valley_idx:peak_idx + 1] - (s[valley_idx:peak_idx + 1] / e_t)
        # Compressive branch (peak -> end & start -> valley): uses E_dyn_comp_half_life
        eps_pl[:valley_idx] = e[:valley_idx] - (s[:valley_idx] / e_c)
        eps_pl[peak_idx + 1:] = e[peak_idx + 1:] - (s[peak_idx + 1:] / e_c)
    else:
        # Compressive branch (peak -> valley): uses E_dyn_comp_half_life
        eps_pl[peak_idx:valley_idx + 1] = e[peak_idx:valley_idx + 1] - (s[peak_idx:valley_idx + 1] / e_c)
        # Tensile branch (start -> peak & valley -> end): uses E_dyn_ten_half_life
        eps_pl[:peak_idx] = e[:peak_idx] - (s[:peak_idx] / e_t)
        eps_pl[valley_idx + 1:] = e[valley_idx + 1:] - (s[valley_idx + 1:] / e_t)

    return float(np.abs(trapz_func(s, eps_pl)))

# Define power-law function for fitting: y = A + B * (x ** C)
def power_func(x, a, b, c):
    return a + b * (x ** c)

timer = BenchmarkTimer()

# Global decision tracking variables
conf_e = "Y"
conf2 = "Y"
conf_range = "N"
choice = "n"
Lim1, Lim2 = 50, 250
sig_val_min, sig_val_max = 50.0, 1200.0
N1_rat_val, N2_rat_val = 100, 500
N1_E10, N2_E10 = 50, 150
N1, N2 = 10, 100
Eps_rate_half_life = 0.0

print("=" * 75)
print("     UNIFIED LCF & RATCHETING TOOL (FABER CA 23109)")
print("=" * 75)

# =============================================================================
# 1. LOAD CONFIGURATION TEMPLATE (JSON)
# =============================================================================
def select_file_dialog(title, filetypes, default_name=""):
    selected_path = ""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        # Ensure dialog window is forced to the front on all OS
        root.attributes('-topmost', True)
        root.lift()
        root.focus_force()
        root.update()
        selected_path = filedialog.askopenfilename(
            parent=root,
            title=title, 
            initialdir=base_dir, 
            filetypes=filetypes
        )
        root.destroy()
    except Exception:
        pass
    if not selected_path and default_name:
        selected_path = input(f"{title} [{default_name}]: ").strip() or default_name
    return os.path.abspath(selected_path) if selected_path else ""

default_cfg_name = "Config_LCF_Campaign.json"
use_cfg_dialog = input("Select existing JSON configuration profile via dialog? (Y/N) [Y]: ").strip().upper() or "Y"
cfg_file_path = select_file_dialog("Select JSON Configuration File", [("JSON files", "*.json")], default_cfg_name) if use_cfg_dialog == "Y" else os.path.join(base_dir, default_cfg_name)

if not os.path.exists(cfg_file_path):
    print(f"Notice: '{os.path.basename(cfg_file_path)}' not found. Creating default configuration template...")
    cfg = {
        "Project_Info": {"Campaign_Name": "LCF Project", "Control_Mode": "Strain-controlled"},
        "CSV_Data_Format": {"Delimiter": ";", "Decimal_Separator": ",", "Column_Names": {"Time": "#knl_A_cas", "Force_or_Stress": "#knl_A_sila", "Deformation_or_Strain": "#knl_A_def", "Cycle": "#knl_A_cykl"}, "Rows_To_Skip_Before_Zero": 1},
        "Material_Strain_Control": {"Name": "AlSi10Mg (SLM Z)", "E0_nominal_MPa": 76000.0},
        "Material_Stress_Control": {"Name": "In718 SLM(Z)", "E0_nominal_MPa": 155000.0},
        "Specimen_Default_Geometry": {"Data_is_direct_stress_strain": False, "Strain_in_percent": False, "D1_outer_mm": 10.94, "D2_inner_mm": 9.04, "Gauge_length_mm": 10.0, "Correction_factor": 1.0},
        "Common_Evaluation_Settings": {"Initial_Young_Modulus_Limits_MPa": {"Bottom": 50.0, "Upper": 250.0}, "Use_Evaluated_E": True, "Elastic_Region_Size_LinReg_MPa": 100.0, "Offset_Plastic_Strain_Rp": 0.0005, "Edyn_Compression_Drop_Percent": 10.0, "Apply_Cyclic_Yield_Stress_Validity_Filter": False, "Cyclic_Yield_Stress_Validity_Range_MPa": [50.0, 1200.0]},
        "Strain_Control_Specific": {"Default_N_Range": [10, 100], "Sig_A_Failure_Drop_Percent": 25.0, "Half_Life_Distortion_Check": True, "Prefer_Previous_Cycle_At_Half_Life": False, "Dynamic_Modulus_Outlier_Filter": {"Tension_E0_ratio": [0.6, 1.3], "Compression_E0_ratio": [0.4, 1.5]}},
        "Stress_Control_Specific": {"Default_N_RAT_Range": [100, 500], "Default_N_E10_Range": [50, 150], "Cyclic_Yield_Stress_Validity_Range_MPa": [50.0, 1200.0], "Representative_Loops_Fractions": [0.166, 0.333, 0.5, 0.666, 0.833]},
        "Protocol_Metadata_Strain_Control": {"Title": "Uniaxial LCF test report", "Specimen_number": "ID11 solid", "Material": "Al5X1 EOS(SLM Z)", "Equipment": "LabControl 100kN/120Nm", "Institution": "VSB-TUO, Czechia", "Department": "Department of Applied Mechanics", "Temperature": "22°C", "Controlled_Signal_Type": "Strain-triangular", "Amplitude": "2.0 %", "Mean_value": "0.0%", "Loading_Rate": "1.0 %/s", "Unloading_Rate": "1.0 %/s", "Frequency_Hz": "0.125 Hz", "Sampling_Frequency": "200 Hz", "Date": "1. 9. 2026", "Dimensions_text": "9.04/10.94 mm", "Type_of_loading": "uniaxial", "Gauge_length_text": "10.0 mm", "Experimenter": "Radim Halama", "Acknowledgments": "SP2026/027", "Note": "Evaluated with FABER LCF Tool."},
        "Protocol_Metadata_Stress_Control": {"Title": "Ratcheting LCF test", "Specimen_number": "ID59 solid", "Material": "In718 SLM(Z)", "Equipment": "Instron 8801", "Institution": "VUT Brno / VSB-TUO, Czechia", "Department": "Department of Applied Mechanics, FME", "Temperature": "22°C", "Controlled_Signal_Type": "Stress-harmonic", "Amplitude": "± 750 MPa", "Mean_value": "+100 MPa", "Loading_Rate": "- kN/s", "Unloading_Rate": "-", "Frequency_Hz": "2.0 Hz", "Sampling_Frequency": "200 points/cycle", "Date": "2024-06-09", "Dimensions_text": "diameter 6.01 mm", "Type_of_loading": "uniaxial", "Gauge_length_text": "12.5 mm", "Experimenter": "Josef Zapletal", "Acknowledgments": "GACR 23-05338S", "Note": "Till fracture"}
    }
    with open(cfg_file_path, 'w', encoding='utf-8') as fw:
        json.dump(cfg, fw, indent=4, ensure_ascii=False)
else:
    try:
        with open(cfg_file_path, 'r', encoding='utf-8') as fr:
            cfg = json.load(fr)
    except json.JSONDecodeError as err:
        print(f"\nERROR: Syntax error in JSON file '{os.path.basename(cfg_file_path)}': {err}")
        sys.exit(1)

# Mode Selection
cfg_mode_str = cfg.get("Project_Info", {}).get("Control_Mode", "Strain-controlled")
default_mode_num = "1" if "strain" in cfg_mode_str.lower() else "2"
prompt_mode = f"\nConfiguration mode: '{cfg_mode_str}'\nSelect test mode:\n  1 = Strain-controlled (LCF)\n  2 = Stress-controlled (Ratcheting)\nChoice [{default_mode_num}]: "
mode_input = input(prompt_mode).strip() or default_mode_num
IS_STRAIN_CONTROL = (mode_input == "1")

# Nominal Young's Modulus E0 Assignment
mat_cfg = cfg.get("Material_Strain_Control" if IS_STRAIN_CONTROL else "Material_Stress_Control", cfg.get("Material", {}))
default_E0 = 76000.0 if IS_STRAIN_CONTROL else 155000.0
material_name = mat_cfg.get("Name", "Alloy")
E0 = float(mat_cfg.get("E0_nominal_MPa", default_E0))
print(f"Active material: '{material_name}' | Reference E0: {E0:.0f} MPa")


# =============================================================================
# CITATION CHECK & STEP-BY-STEP INTERACTIVE PROMPT
# =============================================================================
# Safely detect output folder without raising NameError
target_dir = globals().get('output_dir', locals().get('output_dir', '.'))

bib_files = glob.glob("*.bib")
citation_info = None

# 1. Check if a .bib file exists in the directory
if bib_files:
    print(f"\n--> Found bibliography file: '{bib_files[0]}'")
    use_bib = input(f"  Use '{bib_files[0]}' for protocol citation? (y/n) [y]: ").strip().lower()
    if use_bib != 'n':
        import re
        entry = {}
        try:
            with open(bib_files[0], 'r', encoding='utf-8', errors='ignore') as bf:
                content = bf.read()
            fields = re.findall(r'(\w+)\s*=\s*[\{"]([^\{\}"]+)[\}"]', content, re.IGNORECASE)
            for k, v in fields:
                entry[k.lower()] = v.strip().replace('\n', ' ')
        except Exception:
            pass
            
        if entry:
            author = entry.get('author', '')
            if ' and ' in author:
                authors = author.split(' and ')
                author_str = f"{authors[0].strip()} et al." if len(authors) > 2 else ", ".join([a.strip() for a in authors])
            else:
                author_str = author.strip()

            year = entry.get('year', '')
            title = entry.get('title', '').rstrip('.')
            if len(title) > 75:
                title = title[:72].rstrip() + "..."

            journal = entry.get('journal', entry.get('booktitle', ''))
            volume = entry.get('volume', '')
            pages = entry.get('pages', '')
            raw_doi = entry.get('doi', entry.get('url', '')).strip()
            
            doi_url = f"https://doi.org/{raw_doi.split('doi.org/')[-1].strip()}" if raw_doi else ""
            
            parts = []
            if author_str: parts.append(author_str)
            if year: parts.append(f"({year})")
            if title: parts.append(f'"{title}."')
            if journal:
                j_str = journal
                if volume: j_str += f" {volume}"
                if pages: j_str += f": {pages}"
                parts.append(j_str)
                
            ref_text = " ".join(parts)
            citation_info = {'text': ref_text, 'doi_url': doi_url}
    else:
        bib_files = []

# 2. If no .bib file was found or user declined it, offer manual input
if not bib_files or citation_info is None:
    print("-" * 86)
    print(" No .bib file selected for footer citation.")
    add_cite = input(" Do you want to enter reference details manually? (y/n) [n]: ").strip().lower()
    
    if add_cite == 'y':
        print("\n Enter citation specifications (press Enter to leave empty):")
        author_in  = input("   1. Authors (e.g. Dubey, V.P. et al.): ").strip()
        year_in    = input("   2. Year (e.g. 2025): ").strip()
        title_in   = input("   3. Article Title: ").strip()
        journal_in = input("   4. Journal / Source: ").strip()
        vol_in     = input("   5. Volume / Issue (optional): ").strip()
        pages_in   = input("   6. Pages / Article ID (optional): ").strip()
        doi_input  = input("   7. DOI or URL (optional): ").strip()
        
        ref_parts = []
        if author_in:  ref_parts.append(author_in)
        if year_in:    ref_parts.append(f"({year_in})")
        if title_in:   ref_parts.append(f'"{title_in.rstrip(".")}."')
        if journal_in:
            j_str = journal_in
            if vol_in:   j_str += f" {vol_in}"
            if pages_in: j_str += f": {pages_in}"
            ref_parts.append(j_str)
            
        ref_text = " ".join(ref_parts)
        
        if doi_input:
            doi_url = doi_input if doi_input.startswith("http") else f"https://doi.org/{doi_input.replace('doi.org/', '')}"
        else:
            doi_url = ""
            
        if ref_text or doi_url:
            citation_info = {'text': ref_text, 'doi_url': doi_url}
            print(f"\n  [✓] Reference set: {ref_text}")
        else:
            citation_info = False
    else:
        citation_info = False

# 3. Save citation.txt safely
cite_file_path = os.path.join(target_dir, 'citation.txt')

if citation_info and isinstance(citation_info, dict):
    with open(cite_file_path, 'w', encoding='utf-8') as f:
        f.write(f"{citation_info.get('text', '')}\n{citation_info.get('doi_url', '')}\n")
else:
    if os.path.exists(cite_file_path):
        try:
            os.remove(cite_file_path)
        except OSError:
            pass

print("-" * 86 + "\n")


# =============================================================================
# 2. SELECT RAW DATA CSV FILE, PREVIEW & COLUMN MAPPING
# =============================================================================
print("\n" + "-" * 75)
print("SELECT RAW TEST DATA FILE (.CSV or .TXT)")
default_data_name = 'UNI_Def2d0_ID11_D010926_T120309.csv' if IS_STRAIN_CONTROL else 'AllData.csv'
data_file_path = select_file_dialog("Select Raw Test CSV File", [("CSV and TXT files", "*.csv *.txt"), ("All files", "*.*")], default_data_name)
if not os.path.exists(data_file_path):
    raise FileNotFoundError(f"Data file not found: {data_file_path}")

with open(data_file_path, 'r', encoding='utf-8', errors='ignore') as f:
    sample_lines = [f.readline().strip() for _ in range(3)]

header_line = sample_lines[0] if len(sample_lines) > 0 else ""
data_line_1 = sample_lines[1] if len(sample_lines) > 1 else ""
data_line_2 = sample_lines[2] if len(sample_lines) > 2 else ""

cfg_csv = cfg.get("CSV_Data_Format", {})
auto_delim = ';' if header_line.count(';') >= header_line.count(',') else ','
default_delim = cfg_csv.get("Delimiter", auto_delim)
auto_dec = ',' if (auto_delim == ';' and ',' in data_line_1) else '.'
default_dec = cfg_csv.get("Decimal_Separator", auto_dec)

print("\n" + "=" * 75 + f"\nFILE: {os.path.basename(data_file_path)}\n" + "-" * 75)
print(f"HEADER:      {header_line}\nDATA ROW 1:  {data_line_1}\nDATA ROW 2:  {data_line_2}\n" + "=" * 75)

delimiter = input(f"Enter CSV delimiter [{default_delim}]: ").strip() or default_delim
decimal_sep = input(f"Enter decimal separator [{default_dec}]: ").strip() or default_dec

df_preview = pd.read_csv(data_file_path, delimiter=delimiter, decimal=decimal_sep, nrows=2)
all_columns = list(df_preview.columns)
print("\nDETECTED COLUMNS IN FILE:")
for idx, col in enumerate(all_columns):
    print(f"  [{idx}] {col}")

# Automatic detection with keyword fallback
cfg_cols = cfg_csv.get("Column_Names", {})
keywords_map = {
    'Time': [cfg_cols.get("Time", "").lower(), '#knl_a_cas', '#knl_a_time', '#ch_testtime', 'time (s)', 'time', 'cas', 'čas', 't (s)'],
    'Force': [cfg_cols.get("Force_or_Stress", "").lower(), 'stress (mpa)', '#knl_a_sila', '#knl_a1_sila', '#knl_a_force', 'force (n)', 'load (kn)', 'stress', 'force', 'load', 'sila', 'síla'],
    'Deformation': [cfg_cols.get("Deformation_or_Strain", "").lower(), 'total strain (%)', '#knl_a_def', '#knl_a1_extens', 'total strain', 'strain (%)', 'deformation', 'extens', 'elongation', 'strain', 'def'],
    'Cycle': [cfg_cols.get("Cycle", "").lower(), 'cycle number', '#knl_a_cykl', '#knl_a1_cykl', '#knl_a_cycle', 'cycle', 'cykl', 'cycles']
}

def auto_detect_col(target_key, columns):
    for cand in keywords_map[target_key]:
        if not cand: continue
        for col in columns:
            c_low = col.strip().lower()
            if cand == c_low or cand in c_low:
                if target_key == 'Deformation' and 'plastic' in c_low: continue
                return col
    return columns[0]

def confirm_col(quantity_label, detected_val, columns):
    while True:
        ans = input(f"Confirm column for {quantity_label} [{detected_val}]: ").strip()
        if not ans: return detected_val
        if ans.isdigit() and int(ans) < len(columns): return columns[int(ans)]
        if ans in columns: return ans
        match = [c for c in columns if c.lower() == ans.lower()]
        if match: return match[0]
        print(f"  Warning: Column '{ans}' not recognized. Enter index [0-{len(columns)-1}] or copy name from list above.")

print("\nCONFIRM COLUMN ASSIGNMENTS (Press [Enter] to keep default, or enter index/name):")
col_time = confirm_col("Time", auto_detect_col('Time', all_columns), all_columns)
col_force = confirm_col("Force / Stress", auto_detect_col('Force', all_columns), all_columns)
col_def = confirm_col("Deformation / Strain", auto_detect_col('Deformation', all_columns), all_columns)
col_cycle = confirm_col("Cycle Number", auto_detect_col('Cycle', all_columns), all_columns)

timer.start()

# --- Safe Data Loading and JSON Rows_To_Skip Validation ---
rows_to_skip_json = int(cfg_csv.get("Rows_To_Skip_Before_Zero", 1))
data = pd.read_csv(data_file_path, delimiter=delimiter, decimal=decimal_sep, usecols=[col_time, col_force, col_def, col_cycle])
data = data.rename(columns={col_time: 'Time', col_force: 'Force', col_def: 'Deformation', col_cycle: 'Cycle'})

# Keep a full, untrimmed backup so data can always be recovered
data_raw_untrimmed = data.copy()

# Find approximate first peak force to ensure JSON does not skip past it - CHECK IT!!!
# first_peak_rough_idx = int(data['Force'].abs().iloc[:min(len(data), 5000)].idxmax())

# Safety guard: JSON value cannot exceed total rows or the first loading peak
if rows_to_skip_json >= len(data) - 10:
    print(f"\n[WARNING] JSON 'Rows_To_Skip_Before_Zero' ({rows_to_skip_json}) exceeds file length ({len(data)}). Resetting to 0.")
    rows_to_skip = 0
# elif rows_to_skip_json >= first_peak_rough_idx:
#     print(f"\n[WARNING] JSON 'Rows_To_Skip_Before_Zero' ({rows_to_skip_json}) is past the first peak (row {first_peak_rough_idx})!")
#     print(f"          It would cut away the entire initial loading ramp. Resetting initial skip to 0 for inspection.")
#     rows_to_skip = 0
else:
    rows_to_skip = rows_to_skip_json

# Apply initial safe skip
if rows_to_skip > 0:
    data = data.drop(range(0, rows_to_skip)).reset_index(drop=True)

# Numeric sanitization handling comma/dot decimal strings
for col in ['Force', 'Deformation', 'Time', 'Cycle']:
    if col in data.columns:
        if data[col].dtype == object:
            data[col] = data[col].astype(str).str.replace(',', '.', regex=False)
        data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna(subset=['Force', 'Deformation']).reset_index(drop=True)

# Output directory setup with user customization
data_folder = os.path.dirname(os.path.abspath(data_file_path))
safe_mat_name = material_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_")
mode_tag = "Strain_Control" if IS_STRAIN_CONTROL else "Stress_Control_Ratcheting"
default_folder_name = f"Output_{safe_mat_name}_{mode_tag}"

# Allow user to specify or confirm output folder name
user_folder = input(f"\nEnter output folder name [{default_folder_name}]: ").strip()
folder_name = user_folder if user_folder else default_folder_name

output_dir = os.path.join(data_folder, folder_name)
os.makedirs(output_dir, exist_ok=True)

for search_dir in [data_folder, base_dir]:
    logo_path = os.path.join(search_dir, "Header_logo.png")
    if os.path.exists(logo_path):
        shutil.copy(logo_path, output_dir)
        break

shutil.copy(cfg_file_path, os.path.join(output_dir, "Campaign_Config_Used.json"))
os.chdir(output_dir)

print("\n" + "=" * 75 + f"\n--> ALL FIGURES AND CSVs ARE BEING SAVED TO:\n    {os.path.abspath(output_dir)}\n" + "=" * 75 + "\n")
timer.pause()

# =============================================================================
# 3. SPECIMEN GEOMETRY & COMMON STRESS/STRAIN COMPUTATION
# =============================================================================
cfg_geom = cfg.get("Specimen_Default_Geometry", {})
def_is_direct = "Y" if cfg_geom.get("Data_is_direct_stress_strain", not IS_STRAIN_CONTROL) else "N"
confirmation2 = input(f'\nAre data in the form of force/elongation (Y/N) - if N, stress and strain are expected directly [{ "N" if def_is_direct == "Y" else "Y" }]: ').strip().upper() or ("N" if def_is_direct == "Y" else "Y")

if confirmation2 == 'N':
    S = 1.0
    ini_cycle = 1
    def_pct = "Y" if cfg_geom.get("Strain_in_percent", not IS_STRAIN_CONTROL) else "N"
    conf_strain_pct = input(f'Is strain in percent (%)? (Y/N) [{def_pct}]: ').strip().upper() or def_pct
    gauge_length = 100.0 if conf_strain_pct == 'Y' else 1.0
    correctionFact = float(cfg_geom.get("Correction_factor", 1.0))
else:
    def_D1 = str(cfg_geom.get("D1_outer_mm", 10.94 if IS_STRAIN_CONTROL else 6.30))
    def_D2 = str(cfg_geom.get("D2_inner_mm", 9.04 if IS_STRAIN_CONTROL else 0.0))
    D1 = float(eval(input(f"Enter the Outer diameter of the specimen [{def_D1} mm]: ") or def_D1, {"np": np, "__builtins__": {}}))
    D2 = float(eval(input(f"Enter the Inner diameter of the specimen [{def_D2} mm]: ") or def_D2, {"np": np, "__builtins__": {}}))
    S = math.pi * (D1 * D1 - D2 * D2) / 4.0
    ini_cycle = 0
    def_gl = str(cfg_geom.get("Gauge_length_mm", 10.0))
    gauge_length = float(input(f"Enter extensometer gauge length [{def_gl} mm]: ") or def_gl)
    correctionFact = float(input(f"Enter extensometer correction factor [{cfg_geom.get('Correction_factor', 1.0)}]: ") or str(cfg_geom.get('Correction_factor', 1.0)))

timer.start()

# Handle trailing zeros in cycle column
max_cycle = data['Cycle'].max()
if max_cycle > 0:
    last_valid_idx = data[data['Cycle'] != 0].index.max()
    data.loc[(data.index > last_valid_idx) & (data['Cycle'] == 0), 'Cycle'] = max_cycle

data['Stress'] = data['Force'] / S
data['Deformation'] = data['Deformation'] / correctionFact
data['Strain'] = data['Deformation'] / gauge_length

# Overview Hysteresis Loop
plt.figure()
plt.plot(data.Strain if confirmation2 == 'N' else data.Deformation, data.Stress if confirmation2 == 'N' else data.Force, linestyle='-')
plt.title("Hysteresis Loop - Entire History")
plt.xlabel("Strain" if confirmation2 == 'N' else "Deformation")
plt.ylabel("Stress (MPa)" if confirmation2 == 'N' else "Force (N)")
plt.minorticks_on()
plt.grid(which="major", linestyle=":", color="black")
plt.grid(which="minor", linestyle=":", color="grey")
plt.tight_layout()
plt.savefig('Hysteresis_Loop.png')
timer.pause()
plt.show()
timer.start()

# Fast peak stress extraction via groupby
peaks = data.groupby('Cycle', sort=False)['Stress'].agg(['max', 'min']).reset_index()
upper_peaks = peaks['max'].tolist()
lower_peaks = peaks['min'].tolist()
initial_data = pd.DataFrame({'Cycle': peaks['Cycle'].tolist(), 'Sig_Max': upper_peaks, 'Sig_Min': lower_peaks})
initial_data['Sig_A'] = (initial_data['Sig_Max'] - initial_data['Sig_Min']) * 0.5

plt.figure()
plt.plot(initial_data.Cycle, initial_data.Sig_Max, 'o')
plt.plot(initial_data.Cycle, initial_data.Sig_Min, 'x')
plt.title("SigMax and SigMin vs N")
plt.xlabel("No of Cycles")
plt.ylabel("Stress (MPa)")
plt.minorticks_on()
plt.grid(which="major", linestyle=":", color="black")
plt.grid(which="minor", linestyle=":", color="grey")
plt.tight_layout()
plt.savefig("SigMaxSigMin_vs_N.png")
timer.pause()
plt.show()
timer.start()

plt.figure()
plt.plot(initial_data['Cycle'], initial_data['Sig_A'], marker='o', ls='-', label='Sig_A')
plt.title('Stress Amplitude - strain control' if IS_STRAIN_CONTROL else 'Stress Amplitude - stress control')
plt.xlabel("Number of Cycles")
plt.ylabel("Stress Amplitude (MPa)")
plt.minorticks_on()
plt.grid(which="major", linestyle=":", color="black")
plt.grid(which="minor", linestyle=":", color="grey")
plt.tight_layout()
# plt.savefig('Sig_A.png' if IS_STRAIN_CONTROL else 'amp_mean_stress.png')
timer.pause()
plt.show()

# =============================================================================
# 4a. DETERMINATION OF CYCLES TO FAILURE (Nf)
# =============================================================================
if IS_STRAIN_CONTROL:
    def no_cycles_to_failure():
        global N1, N2
        cfg_strain_n = cfg.get("Strain_Control_Specific", {}).get("Default_N_Range", [10, 100])
        def_n1 = str(cfg_strain_n[0]) if len(cfg_strain_n) > 0 else "10"
        def_n2 = str(cfg_strain_n[1]) if len(cfg_strain_n) > 1 else "100"

        print('\nDefine the region for linear regression (including half life)!')
        N1 = int(input(f"Enter value for N1 [{def_n1}]: ") or def_n1)
        N2 = int(input(f"Enter value for N2 [{def_n2}]: ") or def_n2)

        timer.start()
        ID1 = initial_data['Cycle'].sub(N1).abs().idxmin()
        ID2 = initial_data['Cycle'].sub(N2).abs().idxmin()
        x_init = np.array(initial_data.Cycle[ID1:ID2])
        y_init = np.array(initial_data.Sig_A[ID1:ID2])
        Slope_Nf, Intercept_Nf, R2 = fast_linreg_1d(x_init, y_init)

        initial_data['Sig_A_pred'] = (Slope_Nf * initial_data['Cycle']) + Intercept_Nf
        initial_data['Difference'] = initial_data['Sig_A_pred'] - initial_data['Sig_A']
        initial_data['Percentage'] = 100.0 * (np.abs(initial_data['Difference']) / initial_data['Sig_A'])
        initial_data.to_csv('Upper_peaks.csv')

        decrease_in_pct = float(cfg.get("Strain_Control_Specific", {}).get("Sig_A_Failure_Drop_Percent", 25.0))
        post_N2 = initial_data.drop(range(0, ID2))
        ID_fail = post_N2['Percentage'].sub(decrease_in_pct).abs().idxmin()
        Nf_calc = int(post_N2['Cycle'][ID_fail])
        Sig_A_at_Nf = float(initial_data.loc[initial_data['Cycle'] == Nf_calc, 'Sig_A'].iloc[0])
        print(f'Number of cycles to failure (25% drop in Sig_A): Nf = {Nf_calc}')

        # Plot Nf determination with full history and threshold
        plt.figure(figsize=(10, 6), dpi=120)
        plt.plot(initial_data['Cycle'], initial_data['Sig_A'], label=r'$\sigma_a$ data', color='tab:blue', linewidth=1.8)
        x_trend = np.linspace(initial_data['Cycle'].iloc[ID1], Nf_calc, 100)
        y_trend = Slope_Nf * x_trend + Intercept_Nf
        plt.plot(x_trend, y_trend, linestyle='-.', color='tab:gray', linewidth=2.0,
                 label=f'Linear trend: y={Slope_Nf:.4f}x+{Intercept_Nf:.2f} (R$^2$={R2:.4f})')
        plt.axvline(x=N1, color='tab:green', linestyle=':', linewidth=1.5, label=f'Fit range [$N_1={N1}, N_2={N2}$]')
        plt.axvline(x=N2, color='tab:green', linestyle=':', linewidth=1.5)
        plt.plot(x_trend, y_trend * (1.0 - decrease_in_pct / 100.0), linestyle=':', color='tab:red', linewidth=1.5, label=f'Failure threshold (-{decrease_in_pct:.0f}%)')
        plt.axvline(x=Nf_calc, color='tab:red', linestyle='--', linewidth=2.0, label=f'$N_f = {Nf_calc}$ cycles')
        plt.scatter([Nf_calc], [Sig_A_at_Nf], color='tab:red', s=80, zorder=5)

        plt.annotate(f'$N_f = {Nf_calc}$\n({Sig_A_at_Nf:.1f} MPa)', xy=(Nf_calc, Sig_A_at_Nf),
                     xytext=(Nf_calc * 0.75, Sig_A_at_Nf * 0.88),
                     arrowprops=dict(facecolor='tab:red', arrowstyle='->', lw=1.2),
                     fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='tab:red', alpha=0.9))

        plt.title(f'Determination of Cycles to Failure $N_f$ ({decrease_in_pct:.0f}% Drop in $\sigma_a$)')
        plt.xlabel('Number of Cycles (-)')
        plt.ylabel('Stress Amplitude $\sigma_a$ (MPa)')
        plt.minorticks_on()
        plt.grid(which='major', linestyle=':', color='black', alpha=0.6)
        plt.grid(which='minor', linestyle=':', color='grey', alpha=0.3)
        plt.legend(loc='lower left', framealpha=0.9)
        plt.tight_layout()
        plt.savefig("Nf.png", dpi=300)
        timer.pause()
        plt.show()

        conf = input('Is it OK (Y/N) [Y]: ').strip().upper() or 'Y'
        if conf == 'Y':
            return Nf_calc, Slope_Nf, Intercept_Nf, ID1, ID2
        else:
            return no_cycles_to_failure()

    Nf, Slope_Nf, Intercept_Nf, ID1, ID2 = no_cycles_to_failure()
else:
    Nf = int(data['Cycle'].max())
    print('Number of cycles to failure:', Nf)
    Slope_Nf, Intercept_Nf, ID1, ID2 = 0.0, 0.0, 0, 0

# =============================================================================
# 4b. INSPECTION & INTERACTIVE/AUTOMATIC TRIMMING OF INITIAL ZEROING/TARE DATA
# =============================================================================
print("\n" + "=" * 75)
print("INSPECTION OF INITIAL LOADING DATA (ZEROING / EXTENSOMETER TARE REMOVAL)")
print("=" * 75)

while True:
    first_cyc_num = ini_cycle if ini_cycle in data['Cycle'].values else data['Cycle'].min()
    first_cyc_data = data[data['Cycle'] == first_cyc_num]
    peak_local_idx = first_cyc_data['Stress'].idxmax()

    # Safety check: Is there enough elastic ramp remaining before peak?
    if peak_local_idx < 15:
        print(f"\n[CRITICAL WARNING] Only {peak_local_idx} points remaining before peak stress!")
        print(f"                   Initial trimming was likely too aggressive (skipped {rows_to_skip} rows).")
        restore_ans = input("Do you want to restore full original raw data from CSV? (Y/N) [Y]: ").strip().upper() or "Y"
        if restore_ans == "Y":
            data = data_raw_untrimmed.copy()
            # Recompute Stress and Strain on full data
            data['Stress'] = data['Force'] / S
            data['Deformation'] = data['Deformation'] / correctionFact
            data['Strain'] = data['Deformation'] / gauge_length
            rows_to_skip = 0
            print("[INFO] Original raw dataset restored. Re-evaluating starting ramp...\n")
            continue

    # Show initial loading branch up to peak (+ 10 points after for context)
    view_end_idx = min(len(data), peak_local_idx + 10)
    initial_branch = data.iloc[:view_end_idx].copy()

    # --- Automatic Extensometer Tare / Load Zeroing Jump Detection ---
    suggested_skip = 0
    sub_eps = initial_branch['Strain'].iloc[:peak_local_idx].values
    sub_sig = initial_branch['Stress'].iloc[:peak_local_idx].values

    if len(sub_eps) > 8:
        # Detect sudden step in strain (extensometer tare button press)
        d_eps = np.abs(np.diff(sub_eps))
        med_d_eps = np.median(d_eps)
        eps_scale = med_d_eps if med_d_eps > 1e-7 else (np.std(d_eps) if np.std(d_eps) > 0 else 1e-5)
        tare_jumps = np.where((d_eps / eps_scale >= 8.0) & (d_eps > 1e-4))[0]

        # Detect sudden step in stress (control mode engagement / actuator jog)
        d_sig = np.abs(np.diff(sub_sig))
        med_d_sig = np.median(d_sig)
        sig_scale = med_d_sig if med_d_sig > 1e-3 else (np.std(d_sig) if np.std(d_sig) > 0 else 1.0)
        load_jumps = np.where((d_sig / sig_scale >= 8.0) & (d_sig > 2.0))[0]

        all_candidates = list(tare_jumps) + list(load_jumps)
        if all_candidates:
            last_jump_idx = int(max(all_candidates))
            if last_jump_idx + 1 < peak_local_idx - 5:
                suggested_skip = last_jump_idx + 1

    # --- Plotting Inspection Figures ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=120)

    # Subplot 1: Stress vs Row Index
    ax1.plot(initial_branch.index, initial_branch['Stress'], 'o-', color='tab:blue', markersize=4, label='Stress curve')
    ax1.axvline(x=0, color='tab:gray', linestyle=':', label='Current Start (Row 0)')
    ax1.scatter([peak_local_idx], [data.loc[peak_local_idx, 'Stress']], color='tab:orange', s=80, zorder=5, label=f'Peak (Row {peak_local_idx})')

    if suggested_skip > 0:
        ax1.axvline(x=suggested_skip, color='tab:purple', linestyle='--', linewidth=2.0, 
                    label=f'Auto-detected tare (Row {suggested_skip})')
        ax1.scatter([suggested_skip], [data.loc[suggested_skip, 'Stress']], color='tab:purple', s=70, zorder=6)

    ax1.set_title("Stress vs Row Index")
    ax1.set_xlabel("Row Index in Current Data (-)")
    ax1.set_ylabel("Stress (MPa)")
    ax1.legend(loc='upper left', fontsize=10)
    ax1.minorticks_on()
    ax1.grid(which="major", linestyle=":", color="black", alpha=0.6)

    # Subplot 2: Stress vs Strain
    ax2.plot(initial_branch['Strain'], initial_branch['Stress'], 'o-', color='tab:green', markersize=4, label='Initial branch')
    if suggested_skip > 0:
        ax2.scatter([data.loc[suggested_skip, 'Strain']], [data.loc[suggested_skip, 'Stress']], 
                    color='tab:purple', s=80, zorder=6, label=f'Tare point (Row {suggested_skip})')

    ax2.set_title("Stress vs Strain (Initial Branch)")
    ax2.set_xlabel("Strain (-)")
    ax2.set_ylabel("Stress (MPa)")
    ax2.legend(loc='lower right', fontsize=10)
    ax2.minorticks_on()
    ax2.grid(which="major", linestyle=":", color="black", alpha=0.6)

    plt.tight_layout()
    timer.pause()
    plt.show()
    timer.start()

    print(f"Total skipped ROWS so far: {rows_to_skip} rows from raw CSV (rows_to_skip in JSON)).")
    print(f"Please, interrupt the evaluation and change the value in JSON if initial data is missing!")
    print(f"First stress peak is at row index: {peak_local_idx} ({data.loc[peak_local_idx, 'Stress']:.1f} MPa).")
    
    if suggested_skip > 0:
        print(f"[INFO] Auto-detected tare/zero jump at Row {suggested_skip}.")
    else:
        print("[INFO] No sharp tare jump detected. Loading appears smooth from Row 0.")

    trim_input = input(f"Enter number of initial rows to trim. Write 0 to finish [{suggested_skip}]: ").strip()
    
    chosen_skip = suggested_skip if not trim_input else None

    if chosen_skip is None:
        try:
            chosen_skip = int(trim_input)
        except ValueError:
            print("[ERROR] Please enter a valid integer.\n")
            continue

    if chosen_skip == 0:
        print("[INFO] Current data start confirmed.")
        break
    elif 0 < chosen_skip < peak_local_idx - 3:
        data = data.iloc[chosen_skip:].reset_index(drop=True)
        rows_to_skip += chosen_skip
        print(f"[SUCCESS] Discarded {chosen_skip} rows. Total skipped from raw CSV: {rows_to_skip}.\n")
    else:
        print(f"[WARNING] Invalid count. Must be between 0 and {peak_local_idx - 4}.\n")

# Figure 1: Time History Subplot (Stress vs Time & Strain vs Time) - first 10 cycles
fig_time, (ax_sig_t, ax_eps_t) = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True)
time_vals = data['Time'] if ('Time' in data.columns and data['Time'].notna().any()) else np.arange(len(data))
c_10_match = data[data['Cycle'] <= 10]
idx_10 = c_10_match.index.max() + 1 if not c_10_match.empty else min(len(data), 5000)


ax_sig_t.plot(time_vals[:idx_10], data['Stress'].iloc[:idx_10], label='Axial Stress', color='tab:blue', linewidth=0.8)
ax_sig_t.set_ylabel("Stress (MPa)")
ax_sig_t.set_title("Stress and Strain Time History (First 10 Cycles)")
ax_sig_t.legend(loc="lower right")
ax_sig_t.minorticks_on()
ax_sig_t.grid(which="major", linestyle=":", color="black")
ax_sig_t.grid(which="minor", linestyle=":", color="grey")

ax_eps_t.plot(time_vals[:idx_10], data['Strain'].iloc[:idx_10], label='Axial Strain', color='tab:orange', linewidth=0.8)
ax_eps_t.set_ylabel("Strain (-)")
ax_eps_t.set_xlabel("Time (s)" if 'Time' in data.columns else "Data points (-)")
ax_eps_t.legend(loc="lower right")
ax_eps_t.minorticks_on()
ax_eps_t.grid(which="major", linestyle=":", color="black")
ax_eps_t.grid(which="minor", linestyle=":", color="grey")

plt.tight_layout()
plt.savefig('stress_strain_time_history.png')
timer.pause()
plt.show()
timer.start()

# Figure 2: First 10 Hysteresis Loops

plt.figure()
plt.plot(data['Strain'].iloc[:idx_10], data['Stress'].iloc[:idx_10], label='First 10 cycles', color='tab:blue', linewidth=1.2)
plt.title("Initial Hysteresis Loops (First 10 Cycles)")
plt.xlabel("Strain (-)")
plt.ylabel("Stress (MPa)")
plt.legend(loc="upper left")
plt.minorticks_on()
plt.grid(which="major", linestyle=":", color="black")
plt.grid(which="minor", linestyle=":", color="grey")
plt.tight_layout()
plt.savefig('initial_hysteresis_loops_10_cycles.png')
timer.pause()
plt.show()
timer.start()

# =============================================================================
# 5. INITIAL YOUNG'S MODULUS EVALUATION WITH CONFIRMATION LOOP
# =============================================================================
cfg_common = cfg.get("Common_Evaluation_Settings", {})
def_lim1 = str(int(cfg_common.get("Initial_Young_Modulus_Limits_MPa", {}).get("Bottom", 50)))
def_lim2 = str(int(cfg_common.get("Initial_Young_Modulus_Limits_MPa", {}).get("Upper", 250)))

while True:
    print('\nDetermination of Youngs Modulus')
    print('Define the region for linear regression (Youngs modulus)!')
    Lim1 = int(input(f"Enter value for Limit_bottom (MPa) [{def_lim1}]: ") or def_lim1)
    Lim2 = int(input(f"Enter value for Limit_upper (MPa) [{def_lim2}]: ") or def_lim2)

    timer.start()
    first_cyc_num = ini_cycle if ini_cycle in data['Cycle'].values else data['Cycle'].min()
    first_cyc_data = data[data['Cycle'] == first_cyc_num]
    Max_value = first_cyc_data['Stress'].idxmax()
    res_lower = first_cyc_data.loc[:Max_value, 'Stress'].sub(Lim1).abs().idxmin()
    res_upper = first_cyc_data.loc[:Max_value, 'Stress'].sub(Lim2).abs().idxmin()

    idx_start, idx_end = min(res_lower, res_upper), max(res_lower, res_upper)
    x_e_raw = data.Strain.values[idx_start:idx_end]
    y_e = data.Stress.values[idx_start:idx_end]

    E_calc, E_int, _ = fast_linreg_1d(x_e_raw, y_e)
    shiftE_calc = - E_int / E_calc
    x_e_plot = x_e_raw - shiftE_calc

    print('.............................................')
    print(f'Strain has been shifted by (%): {shiftE_calc * 100:.4f}')
    print(f'Youngs modulus E (MPa): {E_calc:.1f}')

    plt.figure()
    plt.plot(x_e_plot, y_e, 'o')
    plt.title("Young's Modulus Identification")
    plt.plot(x_e_plot, E_calc * x_e_plot, linestyle='--', color='r', label=f'y={E_calc:.4f}*x')
    plt.legend(loc="upper center")
    plt.xlabel("Strain (-)")
    plt.ylabel("Stress (MPa)")
    plt.minorticks_on()
    plt.grid(which="major", linestyle=":", color="black")
    plt.grid(which="minor", linestyle=":", color="grey")
    plt.tight_layout()
    plt.savefig("Young's Modulus.png")
    timer.pause()
    plt.show()

    conf_E_ok = input('Is it OK (Y/N) [Y]: ').strip().upper() or 'Y'
    if conf_E_ok == 'Y':
        timer.start()
        E = E_calc
        shiftE = shiftE_calc
        data['Strain'] = data['Strain'] - shiftE

        timer.pause()
        def_use_eval = "Y" if cfg_common.get("Use_Evaluated_E", True) else "N"
        conf_use_e = input(f'Would you like to use evaluated value {E_calc:.1f} MPa (Y/N/M-modify) - when N, E0={E0:.0f} MPa is used [{def_use_eval}]: ').strip().upper() or def_use_eval
        if conf_use_e == 'N':
            E = E0
        if conf_use_e == 'M':
            E_input = input(f"Young modulus for Wp calculation [{E_calc:.1f}]: ").strip()
            chosen_E = E_calc if not E_input else None

            if chosen_E is None:
                try:
                    chosen_E = int(E_input)
                except ValueError:
                    print("[ERROR] Please enter a valid integer.\n")
                    continue
            E = chosen_E          
        timer.start()
        break

# Setup Evaluation_input arrays
Evaluation_input = np.zeros([6, 1]) if IS_STRAIN_CONTROL else np.zeros([8, 1])
if IS_STRAIN_CONTROL:
    Evaluation_input[0, 0], Evaluation_input[1, 0] = N1, N2
    Evaluation_input[2, 0] = E
    Evaluation_input[5, 0] = E_calc
else:
    Evaluation_input[4, 0] = E_calc
    Evaluation_input[7, 0] = E

data['Elastic_strain'] = data['Stress'] / E
data['Plastic_strain'] = data['Strain'] - data['Elastic_strain']

# =============================================================================
# 6. VECTORIZED PLASTIC WORK (Wp) CALCULATION
# =============================================================================
print('Calculating plastic work (vectorized)...')
Cycles_list = []
Plastic_work = []
stress_arr = data['Stress'].values
plastic_strain_arr = data['Plastic_strain'].values
cycle_arr = data['Cycle'].values

unique_c, u_indices = np.unique(cycle_arr, return_index=True)
u_indices = list(u_indices) + [len(data)]

for idx_c, c_val in enumerate(unique_c):
    Cycles_list.append(c_val)
    c_start = u_indices[idx_c]
    c_end = u_indices[idx_c + 1] - 1
    wp_val = np.abs(trapz_func(stress_arr[c_start:c_end + 1], plastic_strain_arr[c_start:c_end + 1])) if c_end > c_start else 0.0
    Plastic_work.append(float(wp_val))
    if c_val >= (Nf if IS_STRAIN_CONTROL else Nf - 1):
        break

plt.figure()
plt.plot(Cycles_list, Plastic_work, label='$W_P$ per cycle', linestyle='-')
plt.title("Evolution of Plastic Strain Energy Density per Cycle")
plt.legend(loc="upper left")
plt.xlabel("Number of Cycles (-)")
plt.ylabel("$W_P$ (MPa)")
plt.minorticks_on()
plt.grid(which="major", linestyle=":", color="black")
plt.grid(which="minor", linestyle=":", color="grey")
plt.tight_layout()
plt.savefig('WP_vs_N.png')
timer.pause()
plt.show()

# =============================================================================
# 7. DYNAMIC MODULI EVALUATION (CYCLE 2 CALIBRATION + BATCH)
# =============================================================================
strain_arr = data['Strain'].values
target_calib_cycle = unique_c[2] if len(unique_c) > 2 else (unique_c[1] if len(unique_c) > 1 else unique_c[0])
calib_idx = list(unique_c).index(target_calib_cycle)

c_start_cal = u_indices[calib_idx]
c_end_cal = u_indices[calib_idx + 1] - 1
c_stress_cal = stress_arr[c_start_cal:c_end_cal + 1]
c_strain_cal = strain_arr[c_start_cal:c_end_cal + 1]

# --- PART 1: DYNAMIC MODULUS IN TENSION ---
min_s_idx_cal = int(np.argmin(c_stress_cal))
if min_s_idx_cal < len(c_stress_cal) - 3:
    peak_after_cal = min_s_idx_cal + int(np.argmax(c_stress_cal[min_s_idx_cal:]))
    cal_branch_stress = c_stress_cal[min_s_idx_cal : peak_after_cal + 1]
    cal_branch_strain = c_strain_cal[min_s_idx_cal : peak_after_cal + 1]
else:
    peak_idx_cal = int(np.argmax(c_stress_cal))
    cal_branch_stress = c_stress_cal[: peak_idx_cal + 1]
    cal_branch_strain = c_strain_cal[: peak_idx_cal + 1]

def_el_size_t = str(cfg_common.get("Elastic_Region_Size_LinReg_MPa", 100.0 if IS_STRAIN_CONTROL else 160.0))
el_size_val_t = float(def_el_size_t)
skip_pts_t = 0

while True:
    timer.start()
    fit_start_t = min(skip_pts_t, max(0, len(cal_branch_stress) - 3))
    ref_min_stress = cal_branch_stress[fit_start_t]
    search_val_cal_t = ref_min_stress + el_size_val_t
    closest_cal_t = fit_start_t + int(np.argmin(np.abs(cal_branch_stress[fit_start_t:] - search_val_cal_t)))

    fit_end_t = max(fit_start_t + 3, min(closest_cal_t + 1, len(cal_branch_strain)))
    x_fit_cal_t = cal_branch_strain[fit_start_t:fit_end_t]
    y_fit_cal_t = cal_branch_stress[fit_start_t:fit_end_t]
    slope_cal_t, int_cal_t, r2_cal_t = fast_linreg_1d(x_fit_cal_t, y_fit_cal_t)

    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    ax.plot(cal_branch_strain, cal_branch_stress, 'o', color='lightgray', label='Tensile branch (all pts)')
    if fit_start_t > 0:
        ax.plot(cal_branch_strain[:fit_start_t], cal_branch_stress[:fit_start_t], 'x', color='tab:orange', label=f'Skipped corner ({fit_start_t} pts)')
    ax.plot(x_fit_cal_t, y_fit_cal_t, 'o', color='tab:blue', label='Fitted linear region')
    ax.plot(x_fit_cal_t, slope_cal_t * x_fit_cal_t + int_cal_t, '--', color='red', label=f'Fit: E={slope_cal_t:.1f} MPa (R$^2$={r2_cal_t:.4f})')
    ax.set_title(f"Dynamic Modulus in Tension (upper branch) - Calibration (Cycle {target_calib_cycle})")
    ax.set_xlabel("Strain (-)")
    ax.set_ylabel("Stress (MPa)")
    ax.legend(loc="upper left" if not IS_STRAIN_CONTROL else "lower right", fontsize=12)
    ax.minorticks_on()
    ax.grid(which="major", linestyle=":", color="black")
    ax.grid(which="minor", linestyle=":", color="grey")

    # Inset zoom axes (upper left / right)
    axins = ax.inset_axes([0.54, 0.07, 0.44, 0.48] if not IS_STRAIN_CONTROL else [0.06, 0.50, 0.40, 0.44])
    axins.plot(cal_branch_strain, cal_branch_stress, 'o', color='lightgray', markersize=4)
    if fit_start_t > 0:
        axins.plot(cal_branch_strain[:fit_start_t], cal_branch_stress[:fit_start_t], 'x', color='tab:orange', markersize=6)
    axins.plot(x_fit_cal_t, y_fit_cal_t, 'o', color='tab:blue', markersize=5)
    axins.plot(x_fit_cal_t, slope_cal_t * x_fit_cal_t + int_cal_t, '--', color='red', linewidth=1.5)
    z_end_t = min(20, len(cal_branch_strain))
    x_zt, y_zt = cal_branch_strain[:z_end_t], cal_branch_stress[:z_end_t]
    dx_zt, dy_zt = (np.max(x_zt) - np.min(x_zt)) * 0.12, (np.max(y_zt) - np.min(y_zt)) * 0.12
    axins.set_xlim(np.min(x_zt) - dx_zt, np.max(x_zt) + dx_zt)
    axins.set_ylim(np.min(y_zt) - dy_zt, np.max(y_zt) + dy_zt)
    axins.set_title("Zoom: Valley & Fit", fontsize=11, fontweight='bold')
    axins.grid(which="major", linestyle=":", color="black", alpha=0.6)
    axins.tick_params(labelsize=10)
    try: ax.indicate_inset_zoom(axins, edgecolor="0.4")
    except Exception: pass

    plt.tight_layout()
    plt.savefig("Dynamic modulus - Tension - fit calibration cycle.png")
    timer.pause()
    plt.show()

    conf_cal_t = input(f"Is the linear fit in tension (upper) branch on cycle {target_calib_cycle} OK? (Y/N) [Y]: ").strip().upper() or "Y"
    if conf_cal_t == "Y":
        if IS_STRAIN_CONTROL: Evaluation_input[3, 0] = el_size_val_t
        else: Evaluation_input[5, 0] = el_size_val_t
        break
    else:
        print("\nAdjust calibration parameters after viewing the plot:")
        el_size_val_t = float(input(f"Enter elastic region size [{el_size_val_t:.1f} MPa]: ") or str(el_size_val_t))
        skip_pts_t = int(input("Enter initial points to skip at reversal corner [2]: ") or "2")

timer.start()
print('Evaluating dynamic modulus in tensile (upper) branch for all cycles...')
E_dyn_tension_list, sig_int_tension_list = [], []
Cycle_E_dyn = []

for idx_c, c_val in enumerate(unique_c):
    if c_val == (0 if IS_STRAIN_CONTROL else ini_cycle): continue
    c_start, c_end = u_indices[idx_c], u_indices[idx_c + 1] - 1
    c_stress, c_strain = stress_arr[c_start:c_end + 1], strain_arr[c_start:c_end + 1]
    if len(c_stress) < 5: continue

    min_s_idx = int(np.argmin(c_stress if not IS_STRAIN_CONTROL else c_strain))
    if min_s_idx < len(c_stress) - 3:
        peak_after = min_s_idx + int(np.argmax(c_stress[min_s_idx:] if not IS_STRAIN_CONTROL else c_strain[min_s_idx:]))
        t_stress = c_stress[min_s_idx : peak_after + 1]
        t_strain = c_strain[min_s_idx : peak_after + 1]
    else:
        peak_idx = int(np.argmax(c_stress if not IS_STRAIN_CONTROL else c_strain))
        t_stress = c_stress[: peak_idx + 1]
        t_strain = c_strain[: peak_idx + 1]

    f_start = min(skip_pts_t, max(0, len(t_stress) - 3))
    if len(t_stress) > f_start + 2:
        search_val = t_stress[f_start] + el_size_val_t
        closest_idx = f_start + int(np.argmin(np.abs(t_stress[f_start:] - search_val)))
        fit_end = max(f_start + 3, min(closest_idx + 1, len(t_strain)))
        slope_t, int_t, _ = fast_linreg_1d(t_strain[f_start:fit_end], t_stress[f_start:fit_end])
        if slope_t > 0:
            E_dyn_tension_list.append(slope_t)
            sig_int_tension_list.append(int_t)
        else:
            E_dyn_tension_list.append(E_dyn_tension_list[-1] if E_dyn_tension_list else E)
            sig_int_tension_list.append(0.0)
    else:
        E_dyn_tension_list.append(E_dyn_tension_list[-1] if E_dyn_tension_list else E)
        sig_int_tension_list.append(0.0)

    Cycle_E_dyn.append(c_val)
    if c_val >= Nf - 1: break

# Outlier filtering & endpoints
if IS_STRAIN_CONTROL:
    cfg_filt = cfg.get("Strain_Control_Specific", {}).get("Dynamic_Modulus_Outlier_Filter", {})
    r_t = cfg_filt.get("Tension_E0_ratio", [0.6, 1.3])
    s_t = pd.Series(E_dyn_tension_list)
    s_t[(s_t < r_t[0] * E0) | (s_t > r_t[1] * E0)] = np.nan
    E_dyn_tension_list = s_t.interpolate().ffill().bfill().tolist()
    Cycle_E_dyn.append(Nf)
    E_dyn_tension_list.insert(0, E)
    sig_int_tension_list.insert(0, E_int)
    E_dyn_tension_list.append(E_dyn_tension_list[-1])
    sig_int_tension_list.append(sig_int_tension_list[-1])
    Cycle_E_dyn.insert(0, 0)
else:
    E_dyn_tension_list.insert(0, E)
    sig_int_tension_list.insert(0, E_int)
    s_t = pd.Series(E_dyn_tension_list).interpolate().ffill().bfill()
    E_dyn_tension_list = s_t.tolist()

# Plot Tension Dynamic Modulus
min_len_t = min(len(Cycle_E_dyn if IS_STRAIN_CONTROL else Cycles_list), len(E_dyn_tension_list))
plt.figure()
plt.scatter((Cycle_E_dyn if IS_STRAIN_CONTROL else Cycles_list)[:min_len_t], E_dyn_tension_list[:min_len_t], marker='o', label='Dynamic modulus in tension')
plt.title('Dynamic Modulus - Tension (Upper) vs No. of Cycles')
plt.legend(loc='center right')
plt.xlabel("Number of Cycles")
plt.ylabel("Dynamic Modulus - Upper (MPa)")
plt.minorticks_on()
plt.grid(which="major", linestyle=":", color="black")
plt.grid(which="minor", linestyle=":", color="grey")
plt.tight_layout()
plt.savefig("Dynamic modulus-Tension.png")
timer.pause()
plt.show()
timer.start()

# --- PART 2: DYNAMIC MODULUS IN COMPRESSION ---
peak_p_cal = int(np.argmax(c_strain_cal))
if peak_p_cal < len(c_strain_cal) - 3:
    valley_after_cal = peak_p_cal + int(np.argmin(c_strain_cal[peak_p_cal:]))
    cal_c_branch_stress = c_stress_cal[peak_p_cal : valley_after_cal + 1]
    cal_c_branch_strain = c_strain_cal[peak_p_cal : valley_after_cal + 1]
else:
    valley_p_cal = int(np.argmin(c_strain_cal))
    cal_c_branch_stress = c_stress_cal[: valley_p_cal + 1]
    cal_c_branch_strain = c_strain_cal[: valley_p_cal + 1]

def_el_size_c = str(cfg_common.get("Elastic_Region_Size_LinReg_MPa", 100.0 if IS_STRAIN_CONTROL else 160.0))
el_size_val_c = float(def_el_size_c)
skip_pts_c = 0

while True:
    timer.start()
    fit_start_c = min(skip_pts_c, max(0, len(cal_c_branch_stress) - 3))
    ref_max_stress = cal_c_branch_stress[fit_start_c]
    search_val_cal_c = ref_max_stress - el_size_val_c
    closest_cal_c = fit_start_c + int(np.argmin(np.abs(cal_c_branch_stress[fit_start_c:] - search_val_cal_c)))
    fit_end_c = max(fit_start_c + 3, min(closest_cal_c + 1, len(cal_c_branch_strain)))

    x_fit_cal_c = cal_c_branch_strain[fit_start_c:fit_end_c]
    y_fit_cal_c = cal_c_branch_stress[fit_start_c:fit_end_c]
    slope_cal_c, int_cal_c, r2_cal_c = fast_linreg_1d(x_fit_cal_c, y_fit_cal_c)

    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    ax.plot(cal_c_branch_strain, cal_c_branch_stress, 'o', color='lightgray', label='Unloading branch (all pts)')
    if fit_start_c > 0:
        ax.plot(cal_c_branch_strain[:fit_start_c], cal_c_branch_stress[:fit_start_c], 'x', color='tab:orange', label=f'Skipped peak corner ({fit_start_c} pts)')
    ax.plot(x_fit_cal_c, y_fit_cal_c, 'o', color='tab:blue', label='Fitted linear region')
    ax.plot(x_fit_cal_c, slope_cal_c * x_fit_cal_c + int_cal_c, '--', color='red', label=f'Fit: E_comp={slope_cal_c:.1f} MPa (R$^2$={r2_cal_c:.4f})')
    ax.set_title(f"Dynamic Modulus in Compression (bottom) - Calibration (Cycle {target_calib_cycle})")
    ax.set_xlabel("Strain (-)")
    ax.set_ylabel("Stress (MPa)")
    ax.legend(loc="lower right", fontsize=12)
    ax.minorticks_on()
    ax.grid(which="major", linestyle=":", color="black")
    ax.grid(which="minor", linestyle=":", color="grey")

    # Inset zoom axes (lower left / right)
    axins = ax.inset_axes([0.08, 0.48, 0.44, 0.48])
    axins.plot(cal_c_branch_strain, cal_c_branch_stress, 'o', color='lightgray', markersize=4)
    if fit_start_c > 0:
        axins.plot(cal_c_branch_strain[:fit_start_c], cal_c_branch_stress[:fit_start_c], 'x', color='tab:orange', markersize=6)
    axins.plot(x_fit_cal_c, y_fit_cal_c, 'o', color='tab:blue', markersize=5)
    axins.plot(x_fit_cal_c, slope_cal_c * x_fit_cal_c + int_cal_c, '--', color='red', linewidth=1.5)
    z_end_c = min(20, len(cal_c_branch_strain))
    x_zc, y_zc = cal_c_branch_strain[:z_end_c], cal_c_branch_stress[:z_end_c]
    dx_zc, dy_zc = (np.max(x_zc) - np.min(x_zc)) * 0.12, (np.max(y_zc) - np.min(y_zc)) * 0.12
    axins.set_xlim(np.min(x_zc) - dx_zc, np.max(x_zc) + dx_zc)
    axins.set_ylim(np.min(y_zc) - dy_zc, np.max(y_zc) + dy_zc)
    axins.set_title("Zoom: Peak & Fit", fontsize=11, fontweight='bold')
    axins.grid(which="major", linestyle=":", color="black", alpha=0.6)
    axins.tick_params(labelsize=10)
    try: ax.indicate_inset_zoom(axins, edgecolor="0.4")
    except Exception: pass

    plt.tight_layout()
    plt.savefig("Dynamic modulus - Compression - fit calibration cycle.png")
    timer.pause()
    plt.show()

    conf_cal_c = input(f"Is the linear fit in compression (bottom) branch on cycle {target_calib_cycle} OK? (Y/N) [Y]: ").strip().upper() or "Y"
    if conf_cal_c == "Y":
        break
    else:
        print("\nAdjust compression calibration parameters after viewing the plot:")
        el_size_val_c = float(input(f"Enter compressive elastic region size [{el_size_val_c:.1f} MPa]: ") or str(el_size_val_c))
        skip_pts_c = int(input("Enter initial points to skip at peak turnaround corner [2]: ") or "2")

timer.start()
print('Evaluating dynamic modulus in compression (bottom branch of hyst.loop) for all cycles...')
E_dyn_compression_list, sig_int_compression_list = [], []

for idx_c, c_val in enumerate(unique_c):
    if c_val == (0 if IS_STRAIN_CONTROL else ini_cycle): continue
    c_start, c_end = u_indices[idx_c], u_indices[idx_c + 1] - 1
    c_stress, c_strain = stress_arr[c_start:c_end + 1], strain_arr[c_start:c_end + 1]
    if len(c_stress) < 5: continue

    peak_p = int(np.argmax(c_strain))
    if peak_p < len(c_strain) - 3:
        valley_after = peak_p + int(np.argmin(c_strain[peak_p:]))
        b_stress = c_stress[peak_p : valley_after + 1]
        b_strain = c_strain[peak_p : valley_after + 1]
    else:
        valley_p = int(np.argmin(c_strain))
        b_stress = c_stress[: valley_p + 1]
        b_strain = c_strain[: valley_p + 1]

    f_start_c = min(skip_pts_c, max(0, len(b_stress) - 3))
    if len(b_stress) > f_start_c + 2:
        search_val = b_stress[f_start_c] - el_size_val_c
        closest_c = f_start_c + int(np.argmin(np.abs(b_stress[f_start_c:] - search_val)))
        fit_end_c = max(f_start_c + 3, min(closest_c + 1, len(b_strain)))
        slope_c, int_c, _ = fast_linreg_1d(b_strain[f_start_c:fit_end_c], b_stress[f_start_c:fit_end_c])
        if slope_c > 0:
            E_dyn_compression_list.append(slope_c)
            sig_int_compression_list.append(int_c)
        else:
            E_dyn_compression_list.append(E_dyn_compression_list[-1] if E_dyn_compression_list else E)
            sig_int_compression_list.append(0.0)
    else:
        E_dyn_compression_list.append(E_dyn_compression_list[-1] if E_dyn_compression_list else E)
        sig_int_compression_list.append(0.0)

    if c_val >= (Nf - 2 if IS_STRAIN_CONTROL else Nf - 1): break

if IS_STRAIN_CONTROL:
    r_c = cfg_filt.get("Compression_E0_ratio", [0.4, 1.5])
    s_c = pd.Series(E_dyn_compression_list)
    s_c[(s_c < r_c[0] * E0) | (s_c > r_c[1] * E0)] = np.nan
    s_c_clean = s_c.interpolate().ffill().bfill()
    E_dyn_compression_list = (pd.Series([E if E > 0 else E0] * len(s_c)) if s_c_clean.isna().all() else s_c_clean).tolist()
    E_dyn_compression_list.insert(0, E if E > 0 else E0)
    sig_int_compression_list.insert(0, E_int)
    E_dyn_compression_list.extend([E_dyn_compression_list[-1], E_dyn_compression_list[-1]])
    sig_int_compression_list.extend([sig_int_compression_list[-1], sig_int_compression_list[-1]])
else:
    s_c = pd.Series(E_dyn_compression_list).interpolate().ffill().bfill()
    E_dyn_compression_list = (pd.Series([E if E > 0 else E0] * max(1, len(E_dyn_compression_list))) if s_c.isna().all() else s_c).tolist()
    E_dyn_compression_list.insert(0, E if E > 0 else E0)
    sig_int_compression_list.insert(0, E_int)

# Plot Comparison (Tension vs Compression)
min_len_all = min(len(Cycle_E_dyn if IS_STRAIN_CONTROL else Cycles_list), len(E_dyn_tension_list), len(E_dyn_compression_list))
plt.figure()
plt.scatter((Cycle_E_dyn if IS_STRAIN_CONTROL else Cycles_list)[:min_len_all], E_dyn_tension_list[:min_len_all], marker='o', label='Dynamic modulus - upper')
plt.scatter((Cycle_E_dyn if IS_STRAIN_CONTROL else Cycles_list)[:min_len_all], E_dyn_compression_list[:min_len_all], marker='o', label='Dynamic modulus - bottom')
plt.title('Dynamic Modulus Comparison - Upper vs Bottom Branch')
plt.legend(loc="center right")
plt.xlabel("Number of Cycles")
plt.ylabel("Dynamic Modulus (MPa)")
plt.minorticks_on()
plt.grid(which="major", linestyle=":", color="black")
plt.grid(which="minor", linestyle=":", color="grey")
plt.tight_layout()
plt.savefig("Dynamic modulus - comparison.png")
timer.pause()
plt.show()
timer.start()

# =============================================================================
# 8. CYCLIC HARDENING/SOFTENING (AMPLITUDE & MEAN VALUES)
# =============================================================================
c_stats = data[data['Cycle'] <= Nf].groupby('Cycle', sort=False).agg({'Stress': ['max', 'min'], 'Strain': ['max', 'min']})
sigma_A = ((c_stats['Stress']['max'] - c_stats['Stress']['min']) * 0.5).tolist()
sigma_M = ((c_stats['Stress']['max'] + c_stats['Stress']['min']) * 0.5).tolist()
epsilon_A = ((c_stats['Strain']['max'] - c_stats['Strain']['min']) * 0.5).tolist()
epsilon_M = ((c_stats['Strain']['max'] + c_stats['Strain']['min']) * 0.5).tolist()

sigma_M[0], epsilon_M[0] = 0.0, 0.0
if IS_STRAIN_CONTROL:
    sigma_A[0] = initial_data['Sig_Max'].iloc[0]
    epsilon_A[0] = float(data.loc[data['Cycle'] == data['Cycle'].min(), 'Strain'].max())
else:
    first_cyc_idx = data['Cycle'].min()
    sigma_A[0] = float(data.loc[data['Cycle'] == first_cyc_idx, 'Stress'].max()) - (sigma_M[1] if len(sigma_M) > 1 else 0.0)
    epsilon_A[0] = float(data.loc[data['Cycle'] == first_cyc_idx, 'Strain'].max()) - (epsilon_M[1] if len(epsilon_M) > 1 else 0.0)

min_len_sig = min(len(Cycle_E_dyn if IS_STRAIN_CONTROL else Cycles_list), len(sigma_A), len(sigma_M))
plt.figure()
plt.scatter((Cycle_E_dyn if IS_STRAIN_CONTROL else Cycles_list)[:min_len_sig], sigma_A[:min_len_sig], marker='o', ls='-', label='Stress amplitude')
plt.scatter((Cycle_E_dyn if IS_STRAIN_CONTROL else Cycles_list)[:min_len_sig], sigma_M[:min_len_sig], marker='o', ls='-', label='Mean stress')
plt.title('Amplitude-Mean Stress Evolution')
plt.legend(loc='center right')
plt.xlabel("Number of Cycles")
plt.ylabel("Stress Amplitude, Mean Stress (MPa)")
plt.minorticks_on()
plt.grid(which="major", linestyle=":", color="black")
plt.grid(which="minor", linestyle=":", color="grey")
plt.tight_layout()
plt.savefig("Amplitude-mean stress.png")
timer.pause()
plt.show()
timer.start()

min_len_eps = min(len(Cycle_E_dyn if IS_STRAIN_CONTROL else Cycles_list), len(epsilon_A), len(epsilon_M))
plt.figure()
plt.scatter((Cycle_E_dyn if IS_STRAIN_CONTROL else Cycles_list)[:min_len_eps], epsilon_A[:min_len_eps], marker='o', ls='-', label='Strain amplitude')
plt.scatter((Cycle_E_dyn if IS_STRAIN_CONTROL else Cycles_list)[:min_len_eps], epsilon_M[:min_len_eps], marker='o', ls='-', label='Mean strain')
plt.title('Amplitude-Mean Strain Evolution')
plt.legend(loc='center right')
plt.xlabel("Number of Cycles")
plt.ylabel("Strain Parameters (-)")
plt.minorticks_on()
plt.grid(which="major", linestyle=":", color="black")
plt.grid(which="minor", linestyle=":", color="grey")
plt.tight_layout()
plt.savefig("Amplitude-mean strain.png")
timer.pause()
plt.show()
timer.start()

# =============================================================================
# 9. CYCLIC YIELD STRESS EVALUATION (Rp)
# =============================================================================
default_rp = str(cfg_common.get("Offset_Plastic_Strain_Rp", 0.0005))
rp_input_str = input(f"\nEnter the EQUIVALENT tensile offset plastic strain [{default_rp}]: ").strip() or default_rp
rp_equivalent = float(rp_input_str)
if IS_STRAIN_CONTROL: Evaluation_input[4, 0] = rp_equivalent
else: Evaluation_input[6, 0] = rp_equivalent

timer.start()
_L = min(len(Cycles_list), len(sigma_A), len(E_dyn_tension_list), len(E_dyn_compression_list))
output_2 = pd.DataFrame({
    'Cycle': (Cycles_list if IS_STRAIN_CONTROL else Cycles_list)[:_L],
    'sigma_A': sigma_A[:_L], 'sigma_M': sigma_M[:_L], 'epsilon_A': epsilon_A[:_L], 'epsilon_M': epsilon_M[:_L],
    'E_tension': E_dyn_tension_list[:_L], 'E_dyn_tension': E_dyn_tension_list[:_L],
    'E_compression': E_dyn_compression_list[:_L], 'E_dyn_compression': E_dyn_compression_list[:_L]
})
output_2['eps_ap'] = (abs(output_2['epsilon_A'] - 2 * output_2['sigma_A'] / (output_2['E_tension'] + output_2['E_compression'])) if IS_STRAIN_CONTROL
                      else output_2['epsilon_A'] - (output_2['sigma_A'] / E))

if IS_STRAIN_CONTROL:
    plt.figure()
    plt.scatter(Cycles_list[:_L], output_2['eps_ap'], marker='o', ls='-', label='Pl.axial strain ampl.')
    plt.title('Amplitude of Axial Plastic Strain')
    plt.legend(loc='center right')
    plt.xlabel("Number of Cycles")
    plt.ylabel("Amplitude of Axial Plastic Strain (-)")
    plt.minorticks_on()
    plt.grid(which="major", linestyle=":", color="black")
    plt.grid(which="minor", linestyle=":", color="grey")
    plt.tight_layout()
    plt.savefig("Amplitude_axial_plastic_strain.png")
    timer.pause()
    plt.show()
    timer.start()

# Map E_tension to full dataset for yield stress calculation
e_map = dict(zip(output_2['Cycle'], output_2['E_tension']))
data['E_tension'] = data['Cycle'].map(e_map)
data['Eps_pl_calculated'] = data['Strain'] - (data['Stress'] / data['E_tension'])

cyclic_yield_stress, corres_stress_rp0d002 = [], []

for c_val, c_group in data[data['Cycle'] <= (Nf if IS_STRAIN_CONTROL else Nf - 1)].groupby('Cycle', sort=False):
    if len(c_group) < 5 or pd.isna(c_group['E_tension'].iat[0]):
        cyclic_yield_stress.append(np.nan)
        corres_stress_rp0d002.append(np.nan)
        continue

    min_s_idx = c_group['Strain'].idxmin()
    max_s_idx = c_group['Strain'].idxmax()
    branch = c_group.loc[min_s_idx:max_s_idx] if min_s_idx < max_s_idx else pd.concat([c_group.loc[min_s_idx:], c_group.loc[:max_s_idx]])
    offset_eps = branch['Eps_pl_calculated'] - branch.loc[min_s_idx, 'Eps_pl_calculated']

    if offset_eps.dropna().empty:
        cyclic_yield_stress.append(np.nan)
        corres_stress_rp0d002.append(np.nan)
        continue

    idx_rp = (offset_eps - rp_equivalent).abs().idxmin()
    stress_at_rp = c_group.loc[idx_rp, 'Stress']
    min_stress_c = c_group.loc[min_s_idx, 'Stress']
    cyclic_yield_stress.append((stress_at_rp - min_stress_c) * 0.5)
    corres_stress_rp0d002.append(stress_at_rp)

if len(cyclic_yield_stress) > 0 and len(corres_stress_rp0d002) > 0:
    cyclic_yield_stress[0] = corres_stress_rp0d002[0]

output_2['cyclic_yield_stress'] = cyclic_yield_stress[:_L]
output_2['stress_at_rp0.002'] = corres_stress_rp0d002[:_L]
output_2['Plastic_work'] = Plastic_work[:_L]
output_2['Cycles'] = (Cycle_E_dyn if IS_STRAIN_CONTROL else Cycles_list)[:_L]

# Filter & Smooth
output_2['cyclic_yield_stress_NaN'] = np.where(output_2['cyclic_yield_stress'] < 10, np.nan, output_2['cyclic_yield_stress'])
valid_count = output_2['cyclic_yield_stress_NaN'].notna().sum()
interpolated = (output_2['cyclic_yield_stress_NaN'].interpolate(method='polynomial', order=2) if valid_count >= 3
                else output_2['cyclic_yield_stress_NaN'].interpolate(method='linear'))
output_2['cyclic_yield_stress_prefilter'] = interpolated.ffill().bfill()
if output_2['cyclic_yield_stress_prefilter'].isna().all():
    output_2['cyclic_yield_stress_prefilter'] = output_2['cyclic_yield_stress'].fillna(10.0)

# Standalone plot
cyc_col = output_2['Cycles']
plt.figure()
plt.scatter(cyc_col, output_2['cyclic_yield_stress_prefilter'], marker='o', label='Cyclic yield stress')
plt.title('Cyclic yield stress vs No. of cycles')
plt.legend(loc='center right')
plt.xlabel("Number of Cycles")
plt.ylabel("Cyclic Yield Stress (MPa)")
plt.minorticks_on()
plt.grid(which="major", linestyle=":", color="black")
plt.grid(which="minor", linestyle=":", color="grey")
plt.tight_layout()
plt.savefig("Cyclic yield stress.png")
timer.pause()
plt.show()

# Interactive Validity Range Selection
cfg_val_range = cfg.get("Common_Evaluation_Settings", {}).get("Cyclic_Yield_Stress_Validity_Range_MPa", [50.0, 1200.0])
def_apply_val = "Y" if cfg_common.get("Apply_Cyclic_Yield_Stress_Validity_Filter", False) else "N"
conf_range = input(f'\nWould you like to specify a validity range for cyclic yield stress? (Y/N) [{def_apply_val}]: ').strip().upper() or def_apply_val

while conf_range == 'Y':
    y_curr_min, y_curr_max = float(output_2['cyclic_yield_stress_prefilter'].min()), float(output_2['cyclic_yield_stress_prefilter'].max())
    print(f"Current cyclic yield stress values range from {y_curr_min:.1f} MPa to {y_curr_max:.1f} MPa")
    sig_val_min = float(input(f"Enter minimum valid cyclic yield stress (MPa) [{cfg_val_range[0]}]: ") or str(cfg_val_range[0]))
    sig_val_max = float(input(f"Enter maximum valid cyclic yield stress (MPa) [{cfg_val_range[1]}]: ") or str(cfg_val_range[1]))

    timer.start()
    mask_valid = (output_2['cyclic_yield_stress_prefilter'] >= sig_val_min) & (output_2['cyclic_yield_stress_prefilter'] <= sig_val_max)
    s_screened = output_2['cyclic_yield_stress_prefilter'].copy()
    s_screened.loc[~mask_valid] = np.nan
    output_2['cyclic_yield_stress_prefilter'] = s_screened.interpolate(method='linear').ffill().bfill()

    plt.figure()
    plt.scatter(cyc_col, output_2['cyclic_yield_stress_prefilter'], marker='o', label='Cyclic yield stress (screened)')
    plt.title('Cyclic yield stress vs No. of cycles')
    plt.legend(loc='center right')
    plt.xlabel("Number of Cycles")
    plt.ylabel("Cyclic Yield Stress (MPa)")
    plt.minorticks_on()
    plt.grid(which="major", linestyle=":", color="black")
    plt.grid(which="minor", linestyle=":", color="grey")
    plt.tight_layout()
    plt.savefig("Cyclic yield stress.png")
    timer.pause()
    plt.show()

    conf_ok = input('Is this validity range OK? (Y/N) [Y]: ').strip().upper() or 'Y'
    if conf_ok == 'Y':
        break

timer.start()
# 3-curve Comparison Plot required by protocols
plt.figure()
plt.scatter(cyc_col, output_2['sigma_A'], marker='o', ls='-', label='Stress amplitude')
plt.scatter(cyc_col, output_2['sigma_M'], marker='o', ls='-', label='Mean stress')
plt.scatter(cyc_col, output_2['cyclic_yield_stress_prefilter'], color='green', marker='o', label='Cyclic yield stress')
plt.title('Comparison - Cyclic Yield Stress & Stress Amplitude')
plt.legend(loc='center right')
plt.xlabel("Number of Cycles")
plt.ylabel("Stress Parameters (MPa)")
plt.minorticks_on()
plt.grid(which="major", linestyle=":", color="black")
plt.grid(which="minor", linestyle=":", color="grey")
plt.tight_layout()
plt.savefig("Cyclic ye vs Amp Mean stress.png")
timer.pause()
plt.show()
timer.start()

# =============================================================================
# 10. HALF-LIFE CYCLE, HYSTERESIS LOOPS & PLASTIC WORK
# =============================================================================
data['True_stress'] = data['Stress'] * (1 + data['Strain'])
data['True_strain'] = np.log(1 + data['Strain'])
half_life = math.ceil(Nf * 0.5)
half_life_closest = output_2['Cycles'].iloc[(output_2['Cycles'] - half_life).abs().argsort()[:1]]
half_life = int(''.join(map(str, map(int, half_life_closest.tolist()))))

def get_cycle_history(cycle_val):
    c_match = data[data['Cycle'] == cycle_val]
    start_idx, end_idx = c_match.index.min(), c_match.index.max()
    df_hist = pd.DataFrame({'sig_x_true': data['True_stress'].iloc[start_idx:end_idx].values})
    df_hist[['sig_y_true', 'sig_z_true', 'T_xy_true', 'T_xz_true', 'T_yz_true']] = 0
    df_hist['eps_x_true'] = data['True_strain'].iloc[start_idx:end_idx].values
    mu_pl, mu_el = 0.5, 0.3
    df_hist['eps_y_true'] = (((mu_pl - mu_el) * df_hist['sig_x_true'] / E) - (mu_pl * df_hist['eps_x_true']))
    df_hist['eps_z_true'] = df_hist['eps_y_true']
    df_hist[['G_xy_true', 'G_xz_true', 'G_yz_true']] = 0
    return df_hist.reset_index(drop=True)

half_life_history = get_cycle_history(half_life)

if IS_STRAIN_CONTROL:
    unique_cycles = sorted(data['Cycle'].unique())
    idx_u = unique_cycles.index(half_life) if half_life in unique_cycles else 0
    half_life_prev = unique_cycles[idx_u - 1] if idx_u > 0 else half_life
    prev_life_history = get_cycle_history(half_life_prev) if half_life_prev != half_life else None

    plt.figure()
    plt.plot(half_life_history['eps_x_true'], half_life_history['sig_x_true'], linestyle='-', label=f'Target (Cycle {half_life})', color='tab:blue')
    if prev_life_history is not None:
        plt.plot(prev_life_history['eps_x_true'], prev_life_history['sig_x_true'], linestyle='--', label=f'Previous (Cycle {half_life_prev})', color='tab:orange')
    plt.title('Hysteresis Loop Comparison')
    plt.legend(loc='center right')
    plt.xlabel("True strain (-)")
    plt.ylabel("True stress (MPa)")
    plt.minorticks_on()
    plt.grid(which="major", linestyle=":", color="black")
    plt.grid(which="minor", linestyle=":", color="grey")
    plt.tight_layout()
    plt.savefig("HystLoopHalflife_Comparison.png")
    timer.pause()
    plt.show()

    def_prev_choice = "y" if cfg.get("Strain_Control_Specific", {}).get("Prefer_Previous_Cycle_At_Half_Life", False) else "n"
    if prev_life_history is not None and cfg.get("Strain_Control_Specific", {}).get("Half_Life_Distortion_Check", True):
        choice = input(f"\nTarget cycle {half_life} might be distorted. Use previous cycle {half_life_prev} instead? (y/N) [{def_prev_choice}]: ").strip().lower() or def_prev_choice
        if choice == 'y':
            half_life_history = prev_life_history
            half_life = half_life_prev
    timer.start()
else:
    # Representative loops for ratcheting
    sample_cycles = sorted(list(set([1, max(1, Nf//6), max(1, Nf//3), half_life, max(1, 2*Nf//3), max(1, 5*Nf//6), Nf - 1])))
    plt.figure()
    for sc in sample_cycles:
        c_sub = data[data['Cycle'] == sc]
        if len(c_sub) > 0:
            plt.plot(c_sub['Strain'], c_sub['Stress'], label=f'cycle_{sc}')
    plt.title('Evolution of Hysteresis Loops')
    plt.legend(loc='lower right')
    plt.xlabel("Strain (-)")
    plt.ylabel("Stress (MPa)")
    plt.minorticks_on()
    plt.grid(which="major", linestyle=":", color="black")
    plt.grid(which="minor", linestyle=":", color="grey")
    plt.tight_layout()
    plt.savefig("HystLoopsChosen.png")
    timer.pause()
    plt.show()

    plt.figure()
    plt.imshow(plt.imread("Amplitude-mean strain.png"))
    plt.axis("off")  
    plt.show()
    
    # Ratcheting rate evaluation
    cfg_rat_range = cfg.get("Stress_Control_Specific", {}).get("Default_N_RAT_Range", [100, 500])
    while True:
        print('\nPLEASE, DEFINE THE RANGE FOR RATCHETING RATE EVALUATION!')
        N1_rat_val = int(input(f"Enter value for N1_RAT [{cfg_rat_range[0]}]: ") or str(cfg_rat_range[0]))
        N2_rat_val = int(input(f"Enter value for N2_RAT [{cfg_rat_range[1]}]: ") or str(cfg_rat_range[1]))
        timer.start()
        N1_RAT = cyc_col.sub(N1_rat_val).abs().idxmin()
        N2_RAT = cyc_col.sub(N2_rat_val).abs().idxmin()
        x_init_rat = np.array(cyc_col.iloc[N1_RAT:N2_RAT])
        y_init_rat = np.array(output_2['epsilon_M'].iloc[N1_RAT:N2_RAT])
        Eps_rate_half_life, inc_rat, R2_rat = fast_linreg_1d(x_init_rat, y_init_rat)

        plt.figure()
        plt.plot(x_init_rat, y_init_rat, 'o', color='tab:orange', label='Selected mean strain points')
        plt.plot(x_init_rat, Eps_rate_half_life * x_init_rat + inc_rat, linestyle='--', color='r',
                 label=f'y={Eps_rate_half_life:.10f}*x+{inc_rat:.4f}: R$^2$={R2_rat:.5f}')
        plt.title('Ratcheting Rate - Selected Region LinReg')
        plt.xlabel("Number of Cycles")
        plt.ylabel("Mean Strain (-)")
        plt.legend(loc='upper left')
        plt.minorticks_on()
        plt.grid(which="major", linestyle=":", color="black")
        plt.grid(which="minor", linestyle=":", color="grey")
        plt.tight_layout()
        plt.savefig("Ratcheting_rate_regression.png", dpi=200)
        timer.pause()
        plt.show()

        conf_rat = input('Is it OK (Y/N) [Y]: ').strip().upper() or 'Y'
        if conf_rat == 'Y':
            Evaluation_input[0, 0] = N1_rat_val
            Evaluation_input[1, 0] = N2_rat_val
            break

    timer.start()
    # plt.figure()
    # plt.scatter(cyc_col, output_2['epsilon_A'], marker='o', ls='-', label='Strain amplitude')
    # plt.scatter(cyc_col, output_2['epsilon_M'], marker='o', ls='-', label='Mean strain')
    # plt.plot(x_init_rat, Eps_rate_half_life * x_init_rat + inc_rat, linestyle='--', color='r', label=f'y={Eps_rate_half_life:.10f}*x+{inc_rat:.4f}')
    # plt.title('Ratcheting and cyclic hardening/softening curve')
    # plt.legend(loc='upper left')
    # plt.xlabel("Number of cycles")
    # plt.ylabel("Amplitude strain, Mean strain")
    # plt.minorticks_on()
    # plt.grid(which="major", linestyle=":", color="black")
    # plt.grid(which="minor", linestyle=":", color="grey")
    # plt.tight_layout()
    # plt.savefig("cyclic hardening_2.png")
    # timer.pause()
    # plt.show()

half_life_history.to_csv('half_life_history.csv', index=False)

plt.figure()
plt.scatter(half_life_history['eps_x_true'], half_life_history['sig_x_true'], marker='o', ls='-', label=f'Nf/2={half_life}')
plt.title('Hysteresis loop - half life')
plt.legend(loc='center right')
plt.xlabel("True strain (-)")
plt.ylabel("True stress (MPa)")
plt.minorticks_on()
plt.grid(which="major", linestyle=":", color="black")
plt.grid(which="minor", linestyle=":", color="grey")
plt.tight_layout()
plt.savefig("HystLoopHalflife.png")
timer.pause()
plt.show()
timer.start()

# Entire time history & FEM outputs
entire_time_history = pd.DataFrame(data=data['Stress'])
entire_time_history[['sig_y', 'sig_z', 'T_xy', 'T_xz', 'T_yz']] = 0
entire_time_history['eps_x'] = data['Strain']
entire_time_history = entire_time_history.rename(columns={'Stress': 'sig_x'})
mu_pl, mu_el = 0.5, 0.3
entire_time_history['eps_y'] = (((mu_pl - mu_el) * entire_time_history['sig_x'] / E) - (mu_pl * entire_time_history['eps_x']))
entire_time_history['eps_z'] = entire_time_history['eps_y']
entire_time_history[['G_xy', 'G_xz', 'G_yz']] = 0
entire_time_history.to_csv('entire_history.csv')

# =============================================================================
# 11. HALF-LIFE REGRESSIONS, DELTA Wp (VarE) & Nf_10% EVALUATION
# =============================================================================
if not IS_STRAIN_CONTROL:
    cfg_e10_range = cfg.get("Stress_Control_Specific", {}).get("Default_N_E10_Range", [max(1, half_life - 50), min(Nf, half_life + 50)])
    timer.pause()
    print('\nPLEASE, DEFINE THE RANGE FOR HALF-LIFE VALUES EVALUATION (INCLUDING HALF LIFE)!')
    N1_E10 = int(input(f"Enter value for N1 [{cfg_e10_range[0]}]: ") or str(cfg_e10_range[0]))
    N2_E10 = int(input(f"Enter value for N2 [{cfg_e10_range[1]}]: ") or str(cfg_e10_range[1]))
    timer.start()
    ID1 = cyc_col.sub(N1_E10).abs().idxmin()
    ID2 = cyc_col.sub(N2_E10).abs().idxmin()
    Evaluation_input[2, 0] = N1_E10
    Evaluation_input[3, 0] = N2_E10

x_init_hl = np.array(cyc_col.iloc[ID1:ID2])

def get_hl_fit(x_in, y_in):
    y_arr = np.array(y_in, dtype=float)
    valid = ~np.isnan(y_arr)
    if np.sum(valid) >= 2:
        sl, inc, _ = fast_linreg_1d(x_in[valid], y_arr[valid])
        val = sl * half_life + inc
        return (float(val) if not np.isnan(val) else float(np.nanmedian(y_arr[valid]))), (sl, inc)
    fallback = float(np.nanmedian(y_arr)) if np.any(valid) else float(E0)
    return fallback, (0.0, fallback)

Sig_A_half_life = Slope_Nf * half_life + Intercept_Nf if IS_STRAIN_CONTROL else get_hl_fit(x_init_hl, np.array(sigma_A[ID1:ID2]))[0]
Sig_M_half_life, _ = get_hl_fit(x_init_hl, np.array(sigma_M[ID1:ID2]))
Eps_A_half_life, _ = get_hl_fit(x_init_hl, np.array(epsilon_A[ID1:ID2]))
Y_half_life, _ = get_hl_fit(x_init_hl, np.array(output_2['cyclic_yield_stress_prefilter'].iloc[ID1:ID2]))
Wp_half_life, _ = get_hl_fit(x_init_hl, np.array(output_2['Plastic_work'].iloc[ID1:ID2]))
E_dyn_ten_half_life, _ = get_hl_fit(x_init_hl, np.array(E_dyn_tension_list[ID1:ID2]))
E_dyn_comp_half_life, (sl_comp, inc_comp) = get_hl_fit(x_init_hl, np.array(E_dyn_compression_list[ID1:ID2]))

# Plastic strain amplitude at half-life using dynamic moduli
E_dyn_avg_hl = 0.5 * (E_dyn_ten_half_life + E_dyn_comp_half_life)
Eps_Ap_half_life = abs(Eps_A_half_life - (Sig_A_half_life / E_dyn_avg_hl)) if E_dyn_avg_hl > 0 else abs(Eps_A_half_life - (Sig_A_half_life / (E if E > 0 else E0)))

# Plastic work Delta W_p^VarE at half-life using half-life dynamic moduli
c_hl_data = data[data['Cycle'] == half_life]
Wp_vare_half_life = compute_halflife_plastic_work_vare(
    c_hl_data['Stress'].values, c_hl_data['Strain'].values,
    E_dyn_ten_half_life, E_dyn_comp_half_life, (E if E > 0 else E0)
)

Wp_tot = float(output_2['Plastic_work'].sum())
n_wp_cycles = len(output_2['Plastic_work'].dropna())
Wp_mean = Wp_tot / n_wp_cycles if n_wp_cycles > 0 else 0.0


# --- Nf_10% calculation and evaluation ---

# Plot Compression Dynamic Modulus
min_len_c = min(len(Cycle_E_dyn if IS_STRAIN_CONTROL else Cycles_list), len(E_dyn_compression_list))
plt.figure()
plt.scatter((Cycle_E_dyn if IS_STRAIN_CONTROL else Cycles_list)[:min_len_c], E_dyn_compression_list[:min_len_c], marker='o', label='Dynamic modulus in compression (bottom)')
plt.title('Dynamic Modulus - Bottom vs No. of cycles')
plt.legend(loc='center right')
plt.xlabel("Number of Cycles")
plt.ylabel("Dynamic Modulus - Bottom (MPa)")
plt.minorticks_on()
plt.grid(which="major", linestyle=":", color="black")
plt.grid(which="minor", linestyle=":", color="grey")
plt.tight_layout()
plt.savefig("Dynamic modulus-Compression.png")
timer.pause()
plt.show()
timer.start()

decrease_in_pct = float(cfg_common.get("Edyn_Compression_Drop_Percent", 10.0))

# Baseline regression trend for dynamic compression modulus across all cycles
# Uses sl_comp and inc_comp established from steady-state/half-life interval [ID1:ID2]
output_2['E_trend_comp'] = sl_comp * output_2['Cycles'] + inc_comp

# Relative percentage deviation relative to the baseline trendline (+ for drop, - for rise)
output_2['Percentage_Drop'] = 100.0 * (output_2['E_trend_comp'] - output_2['E_compression']) / output_2['E_trend_comp']

# Subset selection after reaching half-life
post_half_life = output_2[output_2['Cycles'] >= half_life]

# Failure condition: absolute deviation from trend >= threshold
failure_cycles = post_half_life[post_half_life['Percentage_Drop'].abs() >= decrease_in_pct]

# Variables to store fit data for final plotting
x_fit_plot, y_fit_plot = None, None
used_approximation_fit = False
zoom_x_lims, zoom_y_lims = None, None
last_cycle = int(output_2['Cycles'].iloc[-1])

# Variables for sparse-data interpolation tracking
used_interpolation = False
interp_segment_x, interp_segment_y = None, None

if not failure_cycles.empty:
    first_fail_idx = failure_cycles.index[0]
    first_fail_row = failure_cycles.loc[first_fail_idx]
    
    # Locate the position of this failing cycle in the output_2 table
    fail_pos = output_2.index.get_loc(first_fail_idx)
    
    if fail_pos > 0:
        prev_row = output_2.iloc[fail_pos - 1]
        n_prev = float(prev_row['Cycles'])
        e_prev = float(prev_row['E_compression'])
        drop_prev = abs(float(prev_row['Percentage_Drop']))
        
        n_fail = float(first_fail_row['Cycles'])
        e_fail = float(first_fail_row['E_compression'])
        drop_fail = abs(float(first_fail_row['Percentage_Drop']))
        
        cycle_gap = int(n_fail - n_prev)
        
        if cycle_gap == 1:
            # Consecutive cycle recording (exact integer cycle)
            Nf_10 = int(n_fail)
            E_comp_at_Nf10 = float(e_fail)
            print(f"Cycles to failure (consecutive cycles recorded): Nf_10 = {Nf_10}")
        else:
            # Sparse logging detected (delta N > 1) -> linear interpolation between bounding cycles
            interp_ratio = (decrease_in_pct - drop_prev) / (drop_fail - drop_prev) if (drop_fail - drop_prev) > 0 else 0.5
            n_calc = n_prev + interp_ratio * (n_fail - n_prev)
            
            Nf_10 = int(round(n_calc))
            E_comp_at_Nf10 = float(e_prev + interp_ratio * (e_fail - e_prev))
            
            used_interpolation = True
            interp_segment_x = [n_prev, n_fail]
            interp_segment_y = [e_prev, e_fail]
            
            print(f"[INFO] Sparse logging detected (gap: {cycle_gap} cycles between cycle {int(n_prev)} and {int(n_fail)}).")
            print(f"       Linear interpolation applied: Nf_10 = {Nf_10} cycles ({E_comp_at_Nf10:.1f} MPa)")
    else:
        # First available point after half-life already exceeded threshold
        Nf_10 = int(first_fail_row['Cycles'])
        E_comp_at_Nf10 = float(first_fail_row['E_compression'])
        print(f"Cycles to failure (criterion reached at boundary): Nf_10 = {Nf_10}")
else:
    print(f"\n[WARNING] Failure criterion (+/-{decrease_in_pct}%) was not reached within data range (max: {last_cycle} cycles).")
    
    use_appr = input("Do you want to apply linear regression / extrapolation for Nf_10? (y/n): ").strip().lower()
    
    if use_appr == 'y':
        fit_successful = False
        while not fit_successful:
            print(f"\nCurrent data range: {int(output_2['Cycles'].iloc[0])} to {last_cycle} cycles.")
            try:
                N1_E10 = float(input("Enter start cycle (N1_E10) for linear fit: "))
                N2_E10 = float(input("Enter end cycle (N2_E10) for linear fit: "))
            except ValueError:
                print("[ERROR] Please enter valid numbers.\n")
                continue
            
            if N1_E10 >= N2_E10:
                print("[ERROR] Start cycle must be strictly smaller than end cycle.\n")
                continue

            # Select data points within user-defined interval
            fit_mask = (output_2['Cycles'] >= N1_E10) & (output_2['Cycles'] <= N2_E10)
            x_fit = output_2.loc[fit_mask, 'Cycles'].values
            y_fit = output_2.loc[fit_mask, 'E_compression'].values
            
            if len(x_fit) >= 2:
                try:
                    # Linear regression of the late-stage failure phase: y = slope * x + intercept
                    slope, intercept = np.polyfit(x_fit, y_fit, 1)
                    
                    # Target the upper or lower inclined threshold line based on slope relative to baseline trend
                    if slope > sl_comp:
                        # Modulus rises faster than baseline trend -> target upper threshold line
                        k_thresh = sl_comp * (1.0 + decrease_in_pct * 0.01)
                        q_thresh = inc_comp * (1.0 + decrease_in_pct * 0.01)
                        direction_label = f"+{decrease_in_pct:.0f}%"
                        print(f"[INFO] Increasing divergence detected (slope: {slope:+.2e}). Targeting UPPER threshold trend ({direction_label}).")
                    else:
                        # Modulus drops faster than baseline trend -> target lower threshold line
                        k_thresh = sl_comp * (1.0 - decrease_in_pct * 0.01)
                        q_thresh = inc_comp * (1.0 - decrease_in_pct * 0.01)
                        direction_label = f"-{decrease_in_pct:.0f}%"
                        print(f"[INFO] Decreasing divergence detected (slope: {slope:+.2e}). Targeting LOWER threshold trend ({direction_label}).")
                    
                    # Analytical intersection between two lines: slope*N + intercept = k_thresh*N + q_thresh
                    delta_slope = slope - k_thresh
                    if abs(delta_slope) < 1e-9:
                        print("[ERROR] Fitted line is parallel to threshold trend. Cannot determine intersection.\n")
                        continue
                    
                    Nf_calc = (q_thresh - intercept) / delta_slope
                    
                    # Ensure intersection lies forward (ahead of fit window start)
                    if Nf_calc < N1_E10:
                        print(f"[ERROR] Calculated intersection ({Nf_calc:.0f}) lies before fit interval start ({N1_E10:.0f}).")
                        print(f"        Trend does not diverge towards the selected threshold. Try another interval.\n")
                        continue
                    
                    temp_Nf_10 = int(round(Nf_calc))
                    temp_E_comp = float(slope * temp_Nf_10 + intercept)
                    
                    # Generate ranges for preview plotting
                    plot_end_x = max(N2_E10, temp_Nf_10, last_cycle) * 1.05
                    temp_x_plot = np.linspace(N1_E10, plot_end_x, 100)
                    temp_y_plot = slope * temp_x_plot + intercept

                    # --- Compute zoom bounds (20 points before N1 + all points after) ---
                    points_prior = output_2[output_2['Cycles'] < N1_E10]
                    if len(points_prior) >= 20:
                        x_min_zoom = float(points_prior['Cycles'].iloc[-20])
                    elif not points_prior.empty:
                        x_min_zoom = float(points_prior['Cycles'].iloc[0])
                    else:
                        x_min_zoom = float(N1_E10)

                    x_max_zoom = float(plot_end_x)
                    x_margin = (x_max_zoom - x_min_zoom) * 0.04
                    cur_xlim = (x_min_zoom - x_margin, x_max_zoom + x_margin)

                    # Threshold lines within preview zoom range
                    x_zoom_trend = np.linspace(cur_xlim[0], cur_xlim[1], 100)
                    y_zoom_target = k_thresh * x_zoom_trend + q_thresh

                    # Compute Y-limits fitting all visible elements
                    pts_in_zoom = output_2[(output_2['Cycles'] >= cur_xlim[0]) & (output_2['Cycles'] <= cur_xlim[1])]
                    y_vals = [temp_E_comp, y_zoom_target[0], y_zoom_target[-1]]
                    if not pts_in_zoom.empty:
                        y_vals.extend(pts_in_zoom['E_compression'].values)
                    
                    y_margin = (max(y_vals) - min(y_vals)) * 0.08
                    cur_ylim = (min(y_vals) - y_margin, max(y_vals) + y_margin)

                except Exception as e:
                    print(f"[ERROR] Linear fitting failed: {e}. Try different interval.\n")
                    continue

                # Interactive zoomed preview plot
                plt.figure(figsize=(9.5, 5.2), dpi=120)
                plt.scatter(output_2['Cycles'], output_2['E_compression'], s=25, alpha=0.7, 
                            label='Measured data', color='tab:blue', edgecolors='none')
                
                # Plot inclined threshold lines
                plt.plot(x_zoom_trend, (sl_comp * x_zoom_trend + inc_comp), linestyle='-.', color='tab:gray', linewidth=1.5, label='Baseline trend')
                plt.plot(x_zoom_trend, y_zoom_target, linestyle=':', color='tab:red', linewidth=1.8, label=f'Target threshold ({direction_label})')
                
                # Plot user linear fit
                plt.plot(temp_x_plot, temp_y_plot, linestyle='--', color='tab:orange', linewidth=2.5, 
                         label=f'Linear fit (k={slope:+.2e})')
                plt.axvline(x=temp_Nf_10, color='tab:red', linestyle='--', linewidth=1.5, label=f'Extrapolated Nf_10 = {temp_Nf_10}')
                plt.scatter([temp_Nf_10], [temp_E_comp], color='tab:red', s=80, zorder=5)

                # Highlight the selected fit interval
                plt.axvspan(N1_E10, N2_E10, color='gold', alpha=0.15, label='Fit interval')

                plt.xlim(cur_xlim)
                plt.ylim(cur_ylim)
                plt.title(f'Zoomed Preview: Extrapolated Nf_10 = {temp_Nf_10} cycles ({direction_label}). Accept?')
                plt.xlabel('Number of cycles (-)')
                plt.ylabel('Dynamic modulus - Compression (MPa)')
                plt.legend(loc='upper left', framealpha=0.9, fontsize=10)
                plt.minorticks_on()
                plt.grid(which='major', linestyle=':', color='black', alpha=0.6)
                plt.grid(which='minor', linestyle=':', color='grey', alpha=0.3)
                plt.tight_layout()
                plt.show()
                
                feedback = input("Is this linear extrapolation OK? (y/n): ").strip().lower()
                if feedback == 'y':
                    Nf_10 = temp_Nf_10
                    E_comp_at_Nf10 = temp_E_comp
                    x_fit_plot = temp_x_plot
                    y_fit_plot = temp_y_plot
                    zoom_x_lims = cur_xlim
                    zoom_y_lims = cur_ylim
                    used_approximation_fit = True
                    fit_successful = True
                    print(f"[SUCCESS] Linear regression accepted: Nf_10 = {Nf_10}")
                else:
                    print("[INFO] Let's try a different interval.\n")
            else:
                print(f"[ERROR] Insufficient points in selected interval (found {len(x_fit)}, at least 2 required). Try again.\n")
    else:
        Nf_10 = last_cycle
        E_comp_at_Nf10 = float(output_2['E_compression'].iloc[-1])
        print(f"Using last solid record: Nf_10 = {Nf_10}")


# --- Final Plotting section ---
plt.figure(figsize=(10, 6), dpi=120)
plt.scatter(output_2['Cycles'], output_2['E_compression'], s=18, alpha=0.7, label='Dynamic modulus in compression', color='tab:blue')

# Generate full x-span for plotting the baseline trend and inclined thresholds
max_x_plot = max(output_2['Cycles'].max(), Nf_10 if Nf_10 is not None else 0) * 1.05
x_trend_full = np.linspace(x_init_hl.min(), max_x_plot, 150)
y_trend_full = sl_comp * x_trend_full + inc_comp

# Inclined threshold boundaries (+/- decrease_in_pct from baseline trend)
y_upper_full = y_trend_full * (1.0 + decrease_in_pct * 0.01)
y_lower_full = y_trend_full * (1.0 - decrease_in_pct * 0.01)

plt.plot(x_trend_full, y_trend_full, linestyle='-.', color='tab:gray', linewidth=2.0, 
         label=f'Baseline trend: y={sl_comp:+.2e}x+{inc_comp:.0f}')
plt.plot(x_trend_full, y_upper_full, linestyle=':', color='tab:red', linewidth=1.5, 
         label=f'Failure threshold ($\pm{decrease_in_pct:.0f}\%$)')
plt.plot(x_trend_full, y_lower_full, linestyle=':', color='tab:red', linewidth=1.5)

plt.axvline(x=half_life, color='tab:green', linestyle=':', linewidth=1.5, label=f'Half-life ($N_f/2 = {half_life}$)')

# Plot linear extrapolation line if applied
if used_approximation_fit and x_fit_plot is not None:
    plt.plot(x_fit_plot, y_fit_plot, linestyle='--', color='tab:orange', linewidth=2.2, label='Linear Extrapolation Fit')

if Nf_10 is not None:
    plt.axvline(x=Nf_10, color='tab:red', linestyle='--', linewidth=2.0, label=f'$N_{{f,10\%}} = {Nf_10}$ cycles')
    plt.scatter([Nf_10], [E_comp_at_Nf10], color='tab:red', s=90, zorder=5)

    xytext_pos = (Nf_10 * 1.05, E_comp_at_Nf10 * 0.98) if Nf_10 < 0.8 * max_x_plot else (Nf_10 * 0.80, E_comp_at_Nf10 * 0.98)

    plt.annotate(f'$N_{{f,10\%}} = {Nf_10}$\n({E_comp_at_Nf10:.1f} MPa)', xy=(Nf_10, E_comp_at_Nf10),
                 xytext=xytext_pos, arrowprops=dict(facecolor='tab:red', arrowstyle='->', lw=1.2),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='tab:red', alpha=0.9))

# Plot interpolation line between bounding recorded cycles if sparse data was detected
if used_interpolation and interp_segment_x is not None:
    plt.plot(interp_segment_x, interp_segment_y, 'o--', color='tab:orange', linewidth=2.0, 
             label=f'Sparse data interpolation [{int(interp_segment_x[0])} - {int(interp_segment_x[1])}]')

title_suffix = ' (Linear Trend & Extrapolation)' if used_approximation_fit else ' (Linear Trend)'
plt.title(f'Determination of Cycles to Failure $N_{{f,10\%}}$ ($\pm{decrease_in_pct:.0f}\%$ from Trend){title_suffix}')
plt.xlabel('Number of Cycles (-)')
plt.ylabel('Dynamic Modulus - Bottom (MPa)')
plt.minorticks_on()
plt.grid(which='major', linestyle=':', color='black', alpha=0.6)
plt.grid(which='minor', linestyle=':', color='grey', alpha=0.3)
plt.legend(loc='lower left', framealpha=0.9, fontsize=10)

if used_approximation_fit:
    plt.xlim(right=max_x_plot)
    y_plot_max = max(output_2['E_compression'].max(), y_upper_full.max(), E_comp_at_Nf10) * 1.02
    plt.ylim(top=y_plot_max)

plt.tight_layout()
plt.savefig('E_compression_Nf10_determination.png', dpi=300)

timer.pause()
plt.show()
timer.start()

comp_time_sec = round(timer.elapsed(), 3)

# =============================================================================
# 12. SAVE TABLES (output_half.csv, Evaluation_input.csv, For_FEM.csv)
# =============================================================================
if IS_STRAIN_CONTROL:
    output_half = pd.DataFrame({
        'Eps_A': [float(Eps_A_half_life)], 'Sig_A': [float(Sig_A_half_life)], 'Nf': [int(Nf)],
        'Nf_E10%': [float(Nf_10) if Nf_10 is not None else np.nan], 'Sig_M': [float(Sig_M_half_life)],
        'E_dyn_tens': [float(E_dyn_ten_half_life)], 'E_dyn_comp': [float(E_dyn_comp_half_life)],
        'Y_cyclic': [float(Y_half_life)], 'DeltaWp': [float(Wp_half_life)],
        'DeltaWp_VarE': [float(Wp_vare_half_life)], 'Wp_tot': [float(Wp_tot)], 'Wp_mean': [float(Wp_mean)],
        'Comp_time_s': [float(comp_time_sec)], 'Eps_Ap (-)': [float(Eps_Ap_half_life)]
    })
    output_3 = pd.DataFrame({'N1': [Evaluation_input[0, 0]], 'N2': [Evaluation_input[1, 0]], 'E': [Evaluation_input[2, 0]],
                             'el.region size for Ycyc': [Evaluation_input[3, 0]], 'offset pl.strain': [Evaluation_input[4, 0]],
                             'E_regr': [Evaluation_input[5, 0]]})
else:
    output_half = pd.DataFrame({
        'Sig_A (MPa)': [float(Sig_A_half_life)], 'Sig_M (MPa)': [float(Sig_M_half_life)], 'Eps_A (-)': [float(Eps_A_half_life)],
        'Eps_rupt (-)': [float(data['Strain'].max())], 'Ratch_rate (-)': [float(Eps_rate_half_life)], 'Nf': [int(Nf)],
        'Nf_E10%': [float(Nf_10) if Nf_10 is not None else np.nan], 'E_dyn_tens (MPa)': [float(E_dyn_ten_half_life)],
        'E_dyn_comp (MPa)': [float(E_dyn_comp_half_life)], 'Y_cyclic (MPa)': [float(Y_half_life)],
        'DeltaWp (MPa)': [float(Wp_half_life)], 'DeltaWp_VarE (MPa)': [float(Wp_vare_half_life)],
        'Wp_tot (MPa)': [float(Wp_tot)], 'Wp_mean (MPa)': [float(Wp_mean)],
        'Comp_time_s': [float(comp_time_sec)], 'Eps_Ap (-)': [float(Eps_Ap_half_life)]
    })
    output_3 = pd.DataFrame({'N1_rat': [Evaluation_input[0, 0]], 'N2_rat': [Evaluation_input[1, 0]], 'N1_E10': [Evaluation_input[2, 0]],
                             'N2_E10': [Evaluation_input[3, 0]], 'E_used': [Evaluation_input[4, 0]], 'E_regression': [Evaluation_input[7, 0]],
                             'el.region size for Ycyc': [Evaluation_input[5, 0]], 'offset pl.strain': [Evaluation_input[6, 0]]})

output_half.to_csv('output_half.csv', index=False, header=True)
output_2.to_csv('amp_mean_values.csv')
output_3.to_csv("Evaluation_input.csv", index=False, header=True)

# Common For_FEM.csv export
c_match = data[data['Cycle'] == Nf]
Nf_cutoff = c_match.index[0] if not c_match.empty else len(data)
For_FEM = pd.DataFrame({
    'Time': data['Time'].iloc[:Nf_cutoff],
    'Strain': data['Strain'].iloc[:Nf_cutoff],
    'Stress': data['Stress'].iloc[:Nf_cutoff],
    'Cycle': data['Cycle'].iloc[:Nf_cutoff]
})
For_FEM.to_csv('For_FEM.csv', index=False)

print("\n" + "=" * 65)
print(f"BENCHMARK: Active CPU computation time: {comp_time_sec:.3f} s")
print("=" * 65)

# =============================================================================
# 13. PROTOCOL GENERATION & TEMPLATE CONFIG EXPORT
# =============================================================================
Protokol = input('\nDo you want to generate the post-processing record protocol (PDF)? (Y/N) [Y]: ').strip().upper() or 'Y'
meta_entered = {}

if Protokol == 'Y':
    if IS_STRAIN_CONTROL:
        import Protocol_LCF
        meta_strain = cfg.get("Protocol_Metadata_Strain_Control", cfg.get("Protocol_Metadata", {}))
        meta_entered = Protocol_LCF.create_Preview_Protocol("Protocol_LCF.pdf", meta=meta_strain, cfg_full=cfg)
    else:
        import Protocol_Ratcheting
        meta_stress = cfg.get("Protocol_Metadata_Stress_Control", cfg.get("Protocol_Metadata", {}))
        meta_entered = Protocol_Ratcheting.create_Preview_Protocol("Protocol_Ratcheting.pdf", meta=meta_stress, cfg_full=cfg)

save_template = input('\nDo you want to save the settings from this run as a new JSON configuration template? (Y/N) [N]: ').strip().upper() or 'N'

if save_template == 'Y':
    default_new_cfg = f"Config_{cfg.get('Material', {}).get('Name', 'Material').replace(' ', '_')}_{'Strain' if IS_STRAIN_CONTROL else 'Stress'}.json"
    new_cfg_name = input(f"Enter filename for the new JSON configuration [{default_new_cfg}]: ").strip() or default_new_cfg
    if not new_cfg_name.lower().endswith('.json'):
        new_cfg_name += '.json'

    default_meta_strain = {"Title": "Uniaxial LCF test report", "Specimen_number": "ID11 solid", "Material": cfg.get("Material", {}).get("Name", "Al5X1"), "Equipment": "LabControl 100kN/120Nm", "Institution": "VSB-TUO, Czechia", "Department": "Department of Applied Mechanics", "Temperature": "22°C", "Controlled_Signal_Type": "Strain-triangular", "Amplitude": "2.0 %", "Mean_value": "0.0%", "Loading_Rate": "1.0 %/s", "Unloading_Rate": "1.0 %/s", "Frequency_Hz": "0.125 Hz", "Sampling_Frequency": "200 Hz", "Date": "1. 9. 2026", "Dimensions_text": "9.04/10.94 mm", "Type_of_loading": "uniaxial", "Gauge_length_text": "10.0 mm", "Experimenter": "Radim Halama", "Acknowledgments": "SP2026/027", "Note": "Evaluated with FABER LCF Tool."}
    default_meta_stress = {"Title": "Ratcheting LCF test", "Specimen_number": "ID59 solid", "Material": cfg.get("Material", {}).get("Name", "In718"), "Equipment": "Instron 8801", "Institution": "VUT Brno / VSB-TUO, Czechia", "Department": "Department of Applied Mechanics, FME", "Temperature": "22°C", "Controlled_Signal_Type": "Stress-harmonic", "Amplitude": "± 750 MPa", "Mean_value": "+100 MPa", "Loading_Rate": "- kN/s", "Unloading_Rate": "-", "Frequency_Hz": "2.0 Hz", "Sampling_Frequency": "200 points/cycle", "Date": "2024-06-09", "Dimensions_text": "diameter 6.01 mm", "Type_of_loading": "uniaxial", "Gauge_length_text": "12.5 mm", "Experimenter": "Josef Zapletal", "Acknowledgments": "GACR 23-05338S", "Note": "Till fracture"}

    final_meta_strain = (meta_entered if meta_entered else cfg.get("Protocol_Metadata_Strain_Control", default_meta_strain)) if IS_STRAIN_CONTROL else cfg.get("Protocol_Metadata_Strain_Control", default_meta_strain)
    final_meta_stress = (meta_entered if meta_entered else cfg.get("Protocol_Metadata_Stress_Control", default_meta_stress)) if not IS_STRAIN_CONTROL else cfg.get("Protocol_Metadata_Stress_Control", default_meta_stress)

    updated_cfg = {
        "Project_Info": {"Campaign_Name": cfg.get("Project_Info", {}).get("Campaign_Name", "LCF Campaign"), "Control_Mode": "Strain-controlled" if IS_STRAIN_CONTROL else "Stress-controlled", "Benchmark_Comp_Time_s": comp_time_sec, "Comment": "Generated automatically from evaluation run."},
        "CSV_Data_Format": {"Delimiter": delimiter, "Decimal_Separator": decimal_sep, "Column_Names": {"Time": col_time, "Force_or_Stress": col_force, "Deformation_or_Strain": col_def, "Cycle": col_cycle}, "Rows_To_Skip_Before_Zero": rows_to_skip},
        "Material_Strain_Control": {"Name": cfg.get("Material_Strain_Control", {}).get("Name", "Al5X1"), "E0_nominal_MPa": float(E) if (IS_STRAIN_CONTROL and 'E' in locals() and E > 0) else float(cfg.get("Material_Strain_Control", {}).get("E0_nominal_MPa", 76000.0))},
        "Material_Stress_Control": {"Name": cfg.get("Material_Stress_Control", {}).get("Name", "In718"), "E0_nominal_MPa": float(E) if (not IS_STRAIN_CONTROL and 'E' in locals() and E > 0) else float(cfg.get("Material_Stress_Control", {}).get("E0_nominal_MPa", 155000.0))},
        "Specimen_Default_Geometry": {"Data_is_direct_stress_strain": (confirmation2 == 'N'), "Strain_in_percent": (locals().get('conf_strain_pct') == 'Y'), "D1_outer_mm": float(locals().get('D1', 10.0)), "D2_inner_mm": float(locals().get('D2', 0.0)), "Gauge_length_mm": float(locals().get('gauge_length', 10.0)), "Correction_factor": float(locals().get('correctionFact', 1.0))},
        "Common_Evaluation_Settings": {
            "Initial_Young_Modulus_Limits_MPa": {"Bottom": float(Lim1), "Upper": float(Lim2)},
            "Use_Evaluated_E": (conf_e == 'Y') if IS_STRAIN_CONTROL else (conf2 == 'Y'),
            "Elastic_Region_Size_LinReg_MPa": float(Evaluation_input[3, 0] if IS_STRAIN_CONTROL else Evaluation_input[5, 0]),
            "Offset_Plastic_Strain_Rp": float(rp_equivalent),
            "Edyn_Compression_Drop_Percent": float(decrease_in_pct),
            "Apply_Cyclic_Yield_Stress_Validity_Filter": (conf_range == 'Y'),
            "Cyclic_Yield_Stress_Validity_Range_MPa": [float(sig_val_min), float(sig_val_max)] if conf_range == 'Y' else cfg.get("Common_Evaluation_Settings", {}).get("Cyclic_Yield_Stress_Validity_Range_MPa", [50.0, 1200.0])
        },
        "Strain_Control_Specific": {
            "Default_N_Range": [int(Evaluation_input[0, 0]), int(Evaluation_input[1, 0])] if IS_STRAIN_CONTROL else cfg.get("Strain_Control_Specific", {}).get("Default_N_Range", [10, 100]),
            "Sig_A_Failure_Drop_Percent": float(cfg.get("Strain_Control_Specific", {}).get("Sig_A_Failure_Drop_Percent", 25.0)),
            "Half_Life_Distortion_Check": cfg.get("Strain_Control_Specific", {}).get("Half_Life_Distortion_Check", True),
            "Prefer_Previous_Cycle_At_Half_Life": (choice == 'y'),
            "Dynamic_Modulus_Outlier_Filter": cfg.get("Strain_Control_Specific", {}).get("Dynamic_Modulus_Outlier_Filter", {"Tension_E0_ratio": [0.6, 1.3], "Compression_E0_ratio": [0.4, 1.5]})
        },
        "Stress_Control_Specific": {
            "Default_N_RAT_Range": [int(N1_rat_val), int(N2_rat_val)],
            "Default_N_E10_Range": [int(N1_E10), int(N2_E10)],
            "Cyclic_Yield_Stress_Validity_Range_MPa": [float(sig_val_min), float(sig_val_max)],
            "Representative_Loops_Fractions": cfg.get("Stress_Control_Specific", {}).get("Representative_Loops_Fractions", [0.166, 0.333, 0.5, 0.666, 0.833])
        },
        "Protocol_Metadata_Strain_Control": final_meta_strain,
        "Protocol_Metadata_Stress_Control": final_meta_stress
    }

    target_json_path = os.path.join(data_folder, new_cfg_name)
    with open(target_json_path, 'w', encoding='utf-8') as fw:
        json.dump(updated_cfg, fw, indent=4, ensure_ascii=False)

    print(f"--> Successfully saved new configuration profile: '{new_cfg_name}' in project directory.")