import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, OptimizeWarning
import sys
import os
import glob
import warnings
import pandas as pd
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import tkinter as tk
from tkinter import filedialog

# Suppress covariance warnings for power-law fits
warnings.filterwarnings('ignore', category=OptimizeWarning)

# =============================================================================
# GLOBAL MATPLOTLIB STYLING MATCHING ORIGINAL STANDARDS
# =============================================================================
plt.rcParams.update({
    'figure.figsize': (10.0, 6.0),
    'agg.path.chunksize': 10000,
    'font.family': 'Times New Roman',
    'figure.dpi': 200,
    'axes.titlesize': 22,
    'axes.labelsize': 22,
    'legend.fontsize': 16,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})

# =============================================================================
# 1. INPUT FILE SELECTION (BROWSE DIRECTORY OR CLI ARGUMENT)
# =============================================================================
print("=" * 86)
print("     LOW-CYCLE FATIGUE EVALUATION - 3D METHOD & DYNAMIC MODULUS EVALUATION     ")
print("=" * 86)

default_input_file = 'Inputs.txt'

if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    input_file_path = sys.argv[1].strip().strip('\'"')
else:
    # Open native file selection window in current directory
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    input_file_path = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Select Input File from Directory",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    root.destroy()

    if not input_file_path:
        input_file_path = default_input_file

if not os.path.exists(input_file_path):
    print(f"\n[ERROR] Input file '{input_file_path}' was not found!")
    sys.exit(1)

print(f"\n--> Selected input file: {input_file_path}")

# =============================================================================
# 2. DATA PARSING (NEW 9-ROW STRUCTURE) & VECTORIZATION
# =============================================================================
with open(input_file_path, 'r', encoding='utf-8') as file:
    raw_lines = [line.strip() for line in file.readlines() if line.strip()]

def parse_line_fast(line_str, dtype=float):
    return np.fromstring(line_str.replace(';', ','), sep=',', dtype=dtype)

# Row 1: Stress amplitudes
stress_vals = parse_line_fast(raw_lines[0])
# Row 2: Strain amplitudes
strain_vals = parse_line_fast(raw_lines[1])
# Row 3: Cycles to failure Nf
cycles_vals = parse_line_fast(raw_lines[2], dtype=int)
# Row 4: Weights
weights_vals = parse_line_fast(raw_lines[3])
# Row 5: Single Young's modulus E (E_const)
E_const = float(raw_lines[4].strip())
# Row 6: Averaged dynamic modulus array (E_var)
young_var_vals = parse_line_fast(raw_lines[5])
# Row 7: Plastic work Wp (read directly from file)
wp_vals = parse_line_fast(raw_lines[6])
# Row 8: Mean stresses
mean_vals = parse_line_fast(raw_lines[7])
# Row 9: Failure criterion string
nf_type = raw_lines[8].strip() if len(raw_lines) > 8 else "Nf_E10"

# =============================================================================
# PARAMETRIC MATERIAL, CASE & OUTPUT DIRECTORY DEFINITION
# =============================================================================
material_name = input("\nEnter Material name [AlSi10MgSLMZ]: ").strip() or "AlSi10MgSLMZ"
case_number   = input("Enter Case number [0]: ").strip() or "0"

safe_mat_name = material_name.replace(" ", "_").replace("/", "_")
output_dir = f"Output_{safe_mat_name}_Case_{case_number}"
os.makedirs(output_dir, exist_ok=True)
print(f"--> All outputs will be saved to: {output_dir}\n")

def out_path(fname):
    return os.path.join(output_dir, fname)

# =============================================================================
# INTERACTIVE METADATA & POST-PROCESSING RECORD
# =============================================================================
print("-" * 86)
print(" METADATA & POST-PROCESSING RECORD (Press Enter to keep default)")
print("-" * 86)

def prompt_meta(field_name, default_val):
    val = input(f"  {field_name} [{default_val}]: ").strip()
    return val if val else default_val

default_samples = str(len(stress_vals))
def_exp_type = f"Uniaxial {material_name}, mean strain 0%, Case {case_number}"
def_note = f"1%/s, Extensometer - gauge length 10 mm. Criterion: {nf_type}"

Experiment_realization = prompt_meta("Experimenter", 'Radim Halama, VSB-TU Ostrava')
Postproces_realization = prompt_meta("Post-processing", 'Radim Halama, VSB-TU Ostrava')
Experiment_type        = prompt_meta("Experiment type / Loading", def_exp_type)
Sample_specifications  = prompt_meta("Sample count / specifications", default_samples)
Date_experiment        = prompt_meta("Date of experiment", '2025')
Date_Postproces        = prompt_meta("Date of post-processing", '13.10.2026')
Note                   = prompt_meta("Notes", def_note)
print("-" * 86 + "\n")

# Build DataFrame
df = pd.DataFrame({
    'Stress': stress_vals,
    'Strain': strain_vals,
    'Cycles': cycles_vals,
    'Weights': weights_vals,
    'Young_var': young_var_vals,
    'W_p': wp_vals,
    'Mean': mean_vals
})

df['E_var_level'] = df.groupby('Strain')['Young_var'].transform('mean')
df['SigMax'] = df['Stress'] + df['Mean']
df['SWT_par'] = df['SigMax'] * df['Strain']

# A) Constant E model (E_const)
df['Elastic_def_const'] = df['Stress'] / E_const
df['Plastic_def_const'] = df['Strain'] - df['Elastic_def_const']
df['Cycles_corr_const'] = df['Cycles']

# B) Variable E model (E_var_level)
df['Elastic_def_var'] = df['Stress'] / df['E_var_level']
df['Plastic_def_var'] = df['Strain'] - df['Elastic_def_var']
df['Cycles_corr_var'] = df['Cycles']

weights_norm = (df['Weights'] / df['Weights'].sum()).values
low_weight_indices = df.index[df['Weights'] < 1.0].to_numpy()
has_low_weight = len(low_weight_indices) > 0

# =============================================================================
# CITATION CHECK & STEP-BY-STEP INTERACTIVE PROMPT
# =============================================================================
bib_files = glob.glob("*.bib")
citation_info = None

if bib_files:
    print(f"--> Found bibliography file: '{bib_files[0]}'")
    use_bib = input(f"  Use '{bib_files[0]}' for citation? (y/n) [y]: ").strip().lower()
    if use_bib != 'n':
        import Protocol
        citation_info = Protocol.get_bib_citation(bib_files[0])
    else:
        bib_files = []

# If no .bib file found or user declined using it
if not bib_files or citation_info is None:
    print("-" * 86)
    print(" No .bib file selected for footer citation.")
    add_cite = input(" Do you want to enter reference details manually? (y/n) [n]: ").strip().lower()
    
    if add_cite == 'y':
        print("\n Enter citation specifications (press Enter to leave an optional field empty):")
        author_in = input("   1. Authors (e.g. Dubey, V.P. et al.): ").strip()
        year_in   = input("   2. Year (e.g. 2025): ").strip()
        title_in  = input("   3. Article Title: ").strip()
        journal_in = input("   4. Journal / Book title: ").strip()
        vol_in    = input("   5. Volume / Issue (optional): ").strip()
        pages_in  = input("   6. Pages / Article ID (optional): ").strip()
        doi_input = input("   7. DOI or URL (optional): ").strip()
        
        # Assemble reference text automatically
        ref_parts = []
        if author_in:
            ref_parts.append(author_in)
        if year_in:
            ref_parts.append(f"({year_in})")
        if title_in:
            ref_parts.append(f'"{title_in.rstrip(".")}."')
        if journal_in:
            j_str = journal_in
            if vol_in:
                j_str += f" {vol_in}"
            if pages_in:
                j_str += f": {pages_in}"
            ref_parts.append(j_str)
            
        ref_text = " ".join(ref_parts)
        
        # Format DOI
        if doi_input:
            doi_url = doi_input if doi_input.startswith("http") else f"https://doi.org/{doi_input.replace('doi.org/', '')}"
            doi_clean = doi_url.split("doi.org/")[-1].strip()
        else:
            doi_clean, doi_url = "", ""
            
        if ref_text or doi_url:
            citation_info = {
                'text': ref_text,
                'doi': doi_clean,
                'doi_url': doi_url
            }
            # Save into output folder
            with open(out_path('citation.txt'), 'w', encoding='utf-8') as f:
                f.write(f"{ref_text}\n{doi_url}\n")
            print(f"\n  [✓] Assembled Reference: {ref_text}")
        else:
            citation_info = False
    else:
        # User explicitly answered NO -> nothing will appear in the protocol!
        citation_info = False
        if os.path.exists(out_path('citation.txt')):
            try:
                os.remove(out_path('citation.txt'))
            except OSError:
                pass

print("-" * 86 + "\n")

# =============================================================================
# 3. REGRESSION & IDENTIFICATION FUNCTIONS
# =============================================================================
def calculate_manual_r2(y_true, y_pred, weights):
    ss_res = np.sum(weights * (y_true - y_pred)**2)
    y_mean = np.average(y_true, weights=weights)
    ss_tot = np.sum(weights * (y_true - y_mean)**2)
    return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

def function_RO(x, a, b):
    return a * (x**b)

def function_MC(x, a, b):
    return a * ((2 * x)**b)

def run_3d_method(stress_log, plastic_log, cycles_log, weights_norm, Young_ref):
    comb_data = np.column_stack((plastic_log, stress_log, cycles_log))
    weighted_mean = np.average(comb_data, axis=0, weights=weights_norm)
    centered = comb_data - weighted_mean
    weighted_centered = centered * np.sqrt(weights_norm[:, np.newaxis])
    
    _, _, vv = np.linalg.svd(weighted_centered)
    dir_vec = vv[0]
    
    n = dir_vec[1] / dir_vec[0]
    c = dir_vec[0] / dir_vec[2]
    b = dir_vec[1] / dir_vec[2]
    K = 10**(weighted_mean[1] - weighted_mean[0] * n)
    Ef = 10**(weighted_mean[0] - weighted_mean[2] * c)
    Sf = 10**(weighted_mean[1] - weighted_mean[2] * b)
    Nt = ((Young_ref * Ef / Sf)**(1.0 / (b - c))) / 2.0
    
    return {
        'comb_data': comb_data, 'vv': vv, 'w_mean': weighted_mean,
        'n': n, 'c': c, 'b': b, 'K': K, 'Ef': Ef, 'Sf': Sf, 'Nt': Nt
    }

def run_2d_regressions(stress_log, plastic_log, cycles_log, weights_norm, Young_ref):
    x_cyc = cycles_log.reshape(-1, 1)
    
    reg_ba = LinearRegression().fit(x_cyc, stress_log, sample_weight=weights_norm)
    b_k = reg_ba.coef_[0]
    Sf_k = 10**reg_ba.intercept_
    R2_BA = reg_ba.score(x_cyc, stress_log, sample_weight=weights_norm)
    
    reg_mc = LinearRegression().fit(x_cyc, plastic_log, sample_weight=weights_norm)
    c_k = reg_mc.coef_[0]
    Ef_k = 10**reg_mc.intercept_
    R2_MC = reg_mc.score(x_cyc, plastic_log, sample_weight=weights_norm)
    
    x_pl = plastic_log.reshape(-1, 1)
    reg_ro = LinearRegression().fit(x_pl, stress_log, sample_weight=weights_norm)
    n_k = reg_ro.coef_[0]
    K_k = 10**reg_ro.intercept_
    R2_RO = reg_ro.score(x_pl, stress_log, sample_weight=weights_norm)
    
    Nt_k = ((Young_ref * Ef_k / Sf_k)**(1.0 / (b_k - c_k))) / 2.0
    
    return {
        'b': b_k, 'Sf': Sf_k, 'R2_BA': R2_BA,
        'c': c_k, 'Ef': Ef_k, 'R2_MC': R2_MC,
        'n': n_k, 'K': K_k, 'R2_RO': R2_RO,
        'Nt': Nt_k
    }

# 1) Fit with E_const
stress_log = np.log10(df['Stress'].values)
cycles_log_const = np.log10(2 * df['Cycles_corr_const'].values)
plastic_log_const = np.log10(df['Plastic_def_const'].values)

res_3D_const = run_3d_method(stress_log, plastic_log_const, cycles_log_const, weights_norm, E_const)
res_2D_const = run_2d_regressions(stress_log, plastic_log_const, cycles_log_const, weights_norm, E_const)

R2_RO_3D_const = calculate_manual_r2(stress_log, np.log10(function_RO(df['Plastic_def_const'].values, res_3D_const['K'], res_3D_const['n'])), weights_norm)
R2_BA_3D_const = calculate_manual_r2(stress_log, np.log10(function_MC(df['Cycles'].values, res_3D_const['Sf'], res_3D_const['b'])), weights_norm)
R2_MC_3D_const = calculate_manual_r2(plastic_log_const, np.log10(function_MC(df['Cycles'].values, res_3D_const['Ef'], res_3D_const['c'])), weights_norm)

# 2) Fit with E_var
cycles_log_var = np.log10(2 * df['Cycles_corr_var'].values)
plastic_log_var = np.log10(df['Plastic_def_var'].values)
E_ref_var = float(df['E_var_level'].mean())

res_3D_var = run_3d_method(stress_log, plastic_log_var, cycles_log_var, weights_norm, E_ref_var)
res_2D_var = run_2d_regressions(stress_log, plastic_log_var, cycles_log_var, weights_norm, E_ref_var)

R2_RO_3D_var = calculate_manual_r2(stress_log, np.log10(function_RO(df['Plastic_def_var'].values, res_3D_var['K'], res_3D_var['n'])), weights_norm)
R2_BA_3D_var = calculate_manual_r2(stress_log, np.log10(function_MC(df['Cycles'].values, res_3D_var['Sf'], res_3D_var['b'])), weights_norm)
R2_MC_3D_var = calculate_manual_r2(plastic_log_var, np.log10(function_MC(df['Cycles'].values, res_3D_var['Ef'], res_3D_var['c'])), weights_norm)

# Save property files directly into the output folder
pd.DataFrame([res_3D_const['n'], res_3D_const['c'], res_3D_const['b'], res_3D_const['K'], res_3D_const['Sf'], res_3D_const['Ef']]).to_csv(out_path('mechprop_3D_const.csv'), index=False, header=False)
pd.DataFrame([res_3D_var['n'], res_3D_var['c'], res_3D_var['b'], res_3D_var['K'], res_3D_var['Sf'], res_3D_var['Ef']]).to_csv(out_path('mechprop_3D_var.csv'), index=False, header=False)
pd.DataFrame([res_2D_const['n'], res_2D_const['c'], res_2D_const['b'], res_2D_const['K'], res_2D_const['Sf'], res_2D_const['Ef']]).to_csv(out_path('mechprop_conv_const.csv'), index=False, header=False)
pd.DataFrame([res_2D_var['n'], res_2D_var['c'], res_2D_var['b'], res_2D_var['K'], res_2D_var['Sf'], res_2D_var['Ef']]).to_csv(out_path('mechprop_conv_var.csv'), index=False, header=False)

reliability_df = pd.DataFrame({
    'CSSC 3D Const': [R2_RO_3D_const], 'Basq 3D Const': [R2_BA_3D_const], 'MC 3D Const': [R2_MC_3D_const],
    'CSSC 3D Var': [R2_RO_3D_var], 'Basq 3D Var': [R2_BA_3D_var], 'MC 3D Var': [R2_MC_3D_var],
    'CSSC Conv Const': [res_2D_const['R2_RO']], 'Basq Conv Const': [res_2D_const['R2_BA']], 'MC Conv Const': [res_2D_const['R2_MC']],
    'CSSC Conv Var': [res_2D_var['R2_RO']], 'Basq Conv Var': [res_2D_var['R2_BA']], 'MC Conv Var': [res_2D_var['R2_MC']]
})
reliability_df.to_csv(out_path('Reliability.csv'), sep=';', index=False, header=True)

with open(out_path('3D_linear_regression.txt'), 'w') as f:
    f.write(f"Directional vector: {res_3D_const['vv'][0]}\n")
    f.write(f"Center of all data points: {res_3D_const['w_mean']}\n")

with open(out_path('Parameters.txt'), 'w') as f:
    f.write(f"Par n: {res_3D_const['n']}\nPar c: {res_3D_const['c']}\nPar b: {res_3D_const['b']}\n")
    f.write(f"Par K: {res_3D_const['K']}\nPar Tauf: {res_3D_const['Sf']}\nPar Gammaf: {res_3D_const['Ef']}\n")
    f.write(f"Nt_3D: {res_3D_const['Nt']}\nNt_2D: {res_2D_const['Nt']}\n")

# =============================================================================
# 4. 4-PARAMETRIC FITTING (SWT & W_p)
# =============================================================================
def function_4_param(x, C1, C2, C3, C4):
    return C1 * (x ** C2) + C3 * (x ** C4)

bounds_4p = ([0.0, -0.35, 0.0, -2.5], [np.inf, 0.0, np.inf, -0.35])

def estimate_p0_4p(x_data, y_data):
    y_min, y_max = np.min(y_data), np.max(y_data)
    x_min, x_max = np.min(x_data), np.max(x_data)
    c1_init = max(y_min * (x_max ** 0.1), 1.0)
    c3_init = max(y_max * (x_min ** 0.6), 1.0)
    return [c1_init, -0.1, c3_init, -0.6]

# Fit SWT
try:
    p0_swt = estimate_p0_4p(df['Cycles'].values, df['SWT_par'].values)
    popt_swt, _ = curve_fit(function_4_param, df['Cycles'].values, df['SWT_par'].values, p0=p0_swt, bounds=bounds_4p, method='trf', maxfev=100000)
    C1_swt, C2_swt, C3_swt, C4_swt = popt_swt
    swt_pred = function_4_param(df['Cycles'].values, *popt_swt)
    r2_swt = calculate_manual_r2(df['SWT_par'].values, swt_pred, weights_norm)
except Exception:
    popt_1p, _ = curve_fit(lambda x, a, b: a * x**b, df['Cycles'], df['SWT_par'], p0=[1000, -0.15])
    C1_swt, C2_swt, C3_swt, C4_swt = popt_1p[0], popt_1p[1], 0.0, -1.0
    swt_pred = function_4_param(df['Cycles'].values, C1_swt, C2_swt, C3_swt, C4_swt)
    r2_swt = calculate_manual_r2(df['SWT_par'].values, swt_pred, weights_norm)

# Fit Wp
try:
    p0_wp = estimate_p0_4p(df['Cycles'].values, df['W_p'].values)
    popt_wp, _ = curve_fit(function_4_param, df['Cycles'].values, df['W_p'].values, p0=p0_wp, bounds=bounds_4p, method='trf', maxfev=100000)
    C1_wp, C2_wp, C3_wp, C4_wp = popt_wp
    wp_pred = function_4_param(df['Cycles'].values, *popt_wp)
    r2_wp = calculate_manual_r2(df['W_p'].values, wp_pred, weights_norm)
except Exception:
    popt_1p, _ = curve_fit(lambda x, a, b: a * x**b, df['Cycles'], df['W_p'], p0=[5000, -0.4])
    C1_wp, C2_wp, C3_wp, C4_wp = popt_1p[0], popt_1p[1], 0.0, -1.0
    wp_pred = function_4_param(df['Cycles'].values, C1_wp, C2_wp, C3_wp, C4_wp)
    r2_wp = calculate_manual_r2(df['W_p'].values, wp_pred, weights_norm)

fatigue_params_4p = pd.DataFrame({
    'Parameter': ['SWT', 'Wp'],
    'C1': [C1_swt, C1_wp],
    'C2': [C2_swt, C2_wp],
    'C3': [C3_swt, C3_wp],
    'C4': [C4_swt, C4_wp],
    'R2': [r2_swt, r2_wp]
})
fatigue_params_4p.to_csv(out_path('Fatigue_Params_4P.csv'), sep=';', index=False)

page2_table_data = pd.DataFrame({
    'ID': np.arange(1, len(df) + 1),
    'Sigma_m': df['Mean'],
    'W_p': df['W_p'],
    'SWT': df['SWT_par']
})
page2_table_data.to_csv(out_path('Page2_Input_Table.csv'), sep=';', index=False)

# =============================================================================
# 5. GENERATE PLOTS (SAVED TO OUTPUT DIRECTORY)
# =============================================================================
cycles_plot = np.logspace(np.log10(max(10, df['Cycles'].min() * 0.7)), np.log10(df['Cycles'].max() * 3.0), 200)

min_p_const, max_p_const = df['Plastic_def_const'].min(), df['Plastic_def_const'].max()
eps_p_plot_const = np.logspace(np.log10(max(1e-6, min_p_const * 0.7)), np.log10(max_p_const * 1.5), 200)

min_p_var, max_p_var = df['Plastic_def_var'].min(), df['Plastic_def_var'].max()
eps_p_plot_var = np.logspace(np.log10(max(1e-6, min_p_var * 0.7)), np.log10(max_p_var * 1.5), 200)

y_ae_3D_const = function_MC(cycles_plot, res_3D_const['Sf'], res_3D_const['b'])
y_ap_3D_const = function_MC(cycles_plot, res_3D_const['Ef'], res_3D_const['c'])
y_ae_3D_var   = function_MC(cycles_plot, res_3D_var['Sf'], res_3D_var['b'])
y_ap_3D_var   = function_MC(cycles_plot, res_3D_var['Ef'], res_3D_var['c'])

# --- PAGE 1: PLOT 1 -> xy_plane.png (CSSC with E_const ONLY, ALL SOLID LINES) ---
plt.figure()
plt.scatter(df['Plastic_def_const'], df['Stress'], color='tab:blue', marker='o', s=60, edgecolors='k', zorder=5, label=r'Data ($E_{\mathrm{const}}$)')
plt.plot(eps_p_plot_const, function_RO(eps_p_plot_const, res_3D_const['K'], res_3D_const['n']), '-', color='tab:blue', linewidth=2.5, label=r'3D ($E_{\mathrm{const}}$)')
plt.plot(eps_p_plot_const, function_RO(eps_p_plot_const, res_2D_const['K'], res_2D_const['n']), '-', color='tab:red', linewidth=1.8, label=rf'Conv. 2D (R²={res_2D_const["R2_RO"]:.4f})')
plt.legend(loc="lower right")
plt.ylabel(r'$\sigma_{\mathrm{a}}$ [MPa]')
plt.xlabel(r'$\varepsilon_{\mathrm{ap}}$ [-]')
# plt.xscale('log')
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.title(r'Cyclic Stress-Strain Curve ($E_{\mathrm{const}}$)')
plt.tight_layout()
plt.savefig(out_path('xy_plane.png'), dpi=300)
plt.close()

# --- PAGE 1: PLOT 2 -> xz_plane.png (CMB with E_const ONLY, ALL SOLID LINES) ---
plt.figure()
plt.scatter(df['Cycles'], df['Elastic_def_const'], color='tab:blue', marker='o', s=50, edgecolors='k', label=r'$\varepsilon_{\mathrm{ae}}$ ($E_{\mathrm{const}}$)')
plt.scatter(df['Cycles'], df['Plastic_def_const'], color='tab:green', marker='s', s=50, edgecolors='k', label=r'$\varepsilon_{\mathrm{ap}}$ ($E_{\mathrm{const}}$)')
plt.scatter(df['Cycles'], df['Strain'], color='tab:red', marker='^', s=60, edgecolors='k', label=r'$\varepsilon_{\mathrm{a}}$ (Total)')

# Solid 3D curves for E_const
plt.plot(cycles_plot, y_ae_3D_const / E_const, color='tab:blue', linestyle='-', linewidth=2.0, label=r'Elastic 3D')
plt.plot(cycles_plot, y_ap_3D_const, color='tab:green', linestyle='-', linewidth=2.0, label=r'Plastic 3D')
plt.plot(cycles_plot, y_ae_3D_const / E_const + y_ap_3D_const, color='tab:red', linestyle='-', linewidth=2.5, label=r'Total 3D')

# Transition lives for 3D and 2D methods
Nt_3D = res_3D_const['Nt']
Nt_2D = res_2D_const['Nt']

plt.vlines(Nt_3D, 0, 0.015, color='tab:red', linestyle=':', linewidth=1.8)
plt.vlines(Nt_2D, 0, 0.015, color='dimgray', linestyle='--', linewidth=1.5)

# Place text annotations close together as a compact block next to the lines:
x_anchor = max(Nt_3D, Nt_2D) * 1.05
y_anchor = 0.0007

plt.text(x_anchor, y_anchor * 1.3, f"$N_{{t,\\mathrm{{3D}}}} = {Nt_3D:.0f}$ cyc",
         fontsize=11, color='tab:red', weight='bold', va='center')
plt.text(x_anchor, y_anchor, f"$N_{{t,\\mathrm{{2D}}}} = {Nt_2D:.0f}$ cyc",
         fontsize=11, color='dimgray', weight='bold', va='center')

plt.legend(loc="lower left", ncol=2)
plt.ylabel(r'$\varepsilon_{\mathrm{ae}}, \varepsilon_{\mathrm{ap}}, \varepsilon_{\mathrm{a}}$ [-]')
plt.xlabel(r'$N_{\mathrm{f}}$ [-]')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.title(r'Coffin-Manson-Basquin Curve ($E_{\mathrm{const}}$)')
plt.tight_layout()
plt.savefig(out_path('xz_plane.png'), dpi=300)
plt.close()

# --- PAGE 1: PLOT 3 -> 3D.png (3D SVD representation) ---
comb_const = res_3D_const['comb_data']
linepts_const = res_3D_const['vv'][0] * np.mgrid[-2:2.5:2j][:, np.newaxis] + res_3D_const['w_mean']

fig_3d = plt.figure(figsize=(10.0, 7.5))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.set_box_aspect([1.1, 1.1, 1.1])
ax_3d.tick_params(axis='x', labelsize=14)
ax_3d.tick_params(axis='y', labelsize=14)
ax_3d.tick_params(axis='z', labelsize=14)
ax_3d.scatter(comb_const[:, 0], comb_const[:, 1], comb_const[:, 2], c='tab:blue', depthshade=False, edgecolors='k', alpha=0.85, s=55, label=r'Data ($E_{\mathrm{const}}$)')
ax_3d.plot(linepts_const[:, 0], linepts_const[:, 1], linepts_const[:, 2], '-', linewidth=2.5, color='tab:blue', label=r'3D Line ($E_{\mathrm{const}}$)')
ax_3d.view_init(elev=22, azim=62)
ax_3d.grid(False)
ax_3d.set_xlabel(r'$\log(\sigma_{\mathrm{a}})$', labelpad=10, fontsize=18)
ax_3d.set_ylabel(r'$\log(\varepsilon_{\mathrm{ap}})$', labelpad=10, fontsize=18)
ax_3d.set_zlabel(r'$\log(2\cdot\mathrm{N_f})$', labelpad=10, fontsize=18)
ax_3d.legend(loc='upper left', fontsize=14)
plt.tight_layout()
plt.savefig(out_path('3D.png'), dpi=300)
plt.close()

# --- STANDALONE 3D COMPARISON FIGURE: E_const vs. E_var (Saved in output folder only) ---
comb_const = res_3D_const['comb_data']
linepts_const = res_3D_const['vv'][0] * np.mgrid[-2:2.5:2j][:, np.newaxis] + res_3D_const['w_mean']

comb_var = res_3D_var['comb_data']
linepts_var = res_3D_var['vv'][0] * np.mgrid[-2:2.5:2j][:, np.newaxis] + res_3D_var['w_mean']

fig_3d_comp = plt.figure(figsize=(10.0, 7.5))
ax_3d_comp = fig_3d_comp.add_subplot(111, projection='3d')
ax_3d_comp.set_box_aspect([1.1, 1.1, 1.1])
ax_3d_comp.tick_params(axis='x', labelsize=14)
ax_3d_comp.tick_params(axis='y', labelsize=14)
ax_3d_comp.tick_params(axis='z', labelsize=14)

# E_const: solid line, filled markers
ax_3d_comp.scatter(comb_const[:, 0], comb_const[:, 1], comb_const[:, 2],
                   c='tab:blue', depthshade=False, edgecolors='k', alpha=0.85, s=55,
                   label=r'Data ($E_{\mathrm{const}}$)')
ax_3d_comp.plot(linepts_const[:, 0], linepts_const[:, 1], linepts_const[:, 2],
                '-', linewidth=2.5, color='tab:blue', label=r'3D Line ($E_{\mathrm{const}}$)')

# E_var: dashed line, empty markers with border only
ax_3d_comp.scatter(comb_var[:, 0], comb_var[:, 1], comb_var[:, 2],
                   facecolors='none', edgecolors='tab:red', depthshade=False,
                   s=65, linewidths=1.6, label=r'Data ($E_{\mathrm{var}}$)')
ax_3d_comp.plot(linepts_var[:, 0], linepts_var[:, 1], linepts_var[:, 2],
                '--', linewidth=2.2, color='tab:red', label=r'3D Line ($E_{\mathrm{var}}$)')

ax_3d_comp.view_init(elev=22, azim=62)
ax_3d_comp.grid(False)
ax_3d_comp.set_xlabel(r'$\log(\varepsilon_{\mathrm{ap}})$', labelpad=10, fontsize=18)
ax_3d_comp.set_ylabel(r'$\log(\sigma_{\mathrm{a}})$', labelpad=10, fontsize=18)
ax_3d_comp.set_zlabel(r'$\log(2\cdot N_{\mathrm{f}})$', labelpad=10, fontsize=18)
ax_3d_comp.legend(loc='upper right')
plt.title(r'3D Method: Comparison $E_{\mathrm{const}}$ vs. $E_{\mathrm{var}}$', fontsize=18, weight='bold')
plt.tight_layout()
plt.savefig(out_path('3D_comparison_Econst_vs_Evar.png'), dpi=300)
plt.close()



# --- PAGE 2: PLOTS ---
# 1. page2_cssc_var.png (CSSC evaluated with E_var) -> Placed TOP-LEFT on Page 2
plt.figure()
plt.scatter(df['Plastic_def_var'], df['Stress'], facecolors='none', edgecolors='tab:blue', marker='o', s=60, linewidths=1.5, zorder=5, label=r'Data ($E_{\mathrm{var}}$)')
plt.plot(eps_p_plot_var, function_RO(eps_p_plot_var, res_3D_var['K'], res_3D_var['n']), '--', color='tab:blue', linewidth=2.5, label=r'3D ($E_{\mathrm{var}}$)')
plt.plot(eps_p_plot_var, function_RO(eps_p_plot_var, res_2D_var['K'], res_2D_var['n']), '--', color='tab:red', linewidth=2.0, label=rf'Conv. 2D (R²={res_2D_var["R2_RO"]:.4f})')
plt.legend(loc="lower right")
plt.ylabel(r'$\sigma_{\mathrm{a}}$ [MPa]')
plt.xlabel(r'$\varepsilon_{\mathrm{ap}}$ [-]')
# plt.xscale('log')
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.title(r'Cyclic Stress-Strain Curve ($E_{\mathrm{var}}$)')
plt.tight_layout()
plt.savefig(out_path('page2_cssc_var.png'), dpi=300)
plt.close()

# 2. page2_cm_comparison.png (Complete CMB Curve: Conv vs Dyn) -> Placed TOP-RIGHT on Page 2
plt.figure()
# E_const (Solid lines, filled markers)
plt.plot(cycles_plot, y_ae_3D_const / E_const, color='tab:blue', linestyle='-', linewidth=2.0)
plt.plot(cycles_plot, y_ap_3D_const, color='tab:green', linestyle='-', linewidth=2.0)
plt.plot(cycles_plot, y_ae_3D_const / E_const + y_ap_3D_const, color='tab:red', linestyle='-', linewidth=2.5)
plt.scatter(df['Cycles'], df['Elastic_def_const'], color='tab:blue', marker='o', s=45, edgecolors='k')
plt.scatter(df['Cycles'], df['Plastic_def_const'], color='tab:green', marker='s', s=45, edgecolors='k')
plt.scatter(df['Cycles'], df['Strain'], color='tab:red', marker='^', s=55, edgecolors='k')

# E_var (Dashed lines, empty markers with border)
plt.plot(cycles_plot, y_ae_3D_var / E_ref_var, color='tab:blue', linestyle='--', linewidth=2.0)
plt.plot(cycles_plot, y_ap_3D_var, color='tab:green', linestyle='--', linewidth=2.0)
plt.plot(cycles_plot, y_ae_3D_var / E_ref_var + y_ap_3D_var, color='tab:red', linestyle='--', linewidth=2.2)
plt.scatter(df['Cycles'], df['Elastic_def_var'], facecolors='none', edgecolors='tab:blue', marker='o', s=50, linewidths=1.5)
plt.scatter(df['Cycles'], df['Plastic_def_var'], facecolors='none', edgecolors='tab:green', marker='s', s=50, linewidths=1.5)
plt.scatter(df['Cycles'], df['Strain'], facecolors='none', edgecolors='tab:red', marker='^', s=60, linewidths=1.5)

legend_elements = [
    Line2D([0], [0], color='tab:blue', linestyle='-', linewidth=2.0,
           marker='o', markerfacecolor='tab:blue', markeredgecolor='k', markersize=7,
           label=r'$\varepsilon_{\mathrm{ae}}$ ($E_{\mathrm{const}}$)'),
    Line2D([0], [0], color='tab:blue', linestyle='--', linewidth=2.0,
           marker='o', markerfacecolor='none', markeredgecolor='tab:blue', markeredgewidth=1.5, markersize=7,
           label=r'$\varepsilon_{\mathrm{ae}}$ ($E_{\mathrm{var}}$)'),

    Line2D([0], [0], color='tab:green', linestyle='-', linewidth=2.0,
           marker='s', markerfacecolor='tab:green', markeredgecolor='k', markersize=7,
           label=r'$\varepsilon_{\mathrm{ap}}$ ($E_{\mathrm{const}}$)'),
    Line2D([0], [0], color='tab:green', linestyle='--', linewidth=2.0,
           marker='s', markerfacecolor='none', markeredgecolor='tab:green', markeredgewidth=1.5, markersize=7,
           label=r'$\varepsilon_{\mathrm{ap}}$ ($E_{\mathrm{var}}$)'),

    Line2D([0], [0], color='tab:red', linestyle='-', linewidth=2.5,
           marker='^', markerfacecolor='tab:red', markeredgecolor='k', markersize=8,
           label=r'$\varepsilon_{\mathrm{a}}$ ($E_{\mathrm{const}}$)'),
    Line2D([0], [0], color='tab:red', linestyle='--', linewidth=2.2,
           marker='^', markerfacecolor='none', markeredgecolor='tab:red', markeredgewidth=1.5, markersize=8,
           label=r'$\varepsilon_{\mathrm{a}}$ ($E_{\mathrm{var}}$)')
]

plt.xlabel(r'$N_f$ [-]')
plt.ylabel(r'Strain Amplitudes $\varepsilon_{\mathrm{ae}}, \varepsilon_{\mathrm{ap}}, \varepsilon_{\mathrm{a}}$ [-]')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.legend(handles=legend_elements, loc='lower left', ncol=3, fontsize=8)
plt.title(r'Complete CMB: $E_{\mathrm{const}}$ vs. $E_{\mathrm{var}}$')
plt.tight_layout()
plt.savefig(out_path('page2_cm_comparison.png'), dpi=300)
plt.close()

# 3. page2_energy_plots.png (SWT and Wp in one plot) -> Placed MIDDLE-RIGHT on Page 2
plt.figure()
x_smooth = np.logspace(np.log10(df['Cycles'].min() * 0.8), np.log10(df['Cycles'].max() * 1.5), 100)
plt.scatter(df['Cycles'], df['SWT_par'], color='tab:red', marker='o', s=60, edgecolors='k', zorder=5, label=r'SWT Data')
plt.plot(x_smooth, function_4_param(x_smooth, *popt_swt), '-', color='tab:red', linewidth=2.5, label=r'SWT 4-Param Fit')
plt.scatter(df['Cycles'], df['W_p'], color='tab:blue', marker='s', s=60, edgecolors='k', zorder=5, label=r'$W_p$ Data')
plt.plot(x_smooth, function_4_param(x_smooth, *popt_wp), '--', color='tab:blue', linewidth=2.5, label=r'$W_p$ 4-Param Fit')
plt.ylabel(r'$\mathrm{SWT},\, W_p$ [MPa]')
plt.xlabel(r'Number of Cycles $N_f$ [-]')
# plt.xscale('log')
# plt.yscale('log')
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.legend(loc='upper right', framealpha=0.9)
plt.title(r'SWT & $W_p$ vs. $N_f$')
plt.tight_layout()
plt.savefig(out_path('page2_energy_plots.png'), dpi=300)
plt.close()

# =============================================================================
# 6. EXPORT METADATA & EXECUTE PROTOCOL
# =============================================================================
input_record = [
    Experiment_realization, Postproces_realization, Experiment_type,
    Sample_specifications, Date_experiment, Date_Postproces, Note, 
    res_3D_const['Nt'], input_file_path, nf_type, res_2D_const['Nt']
]
pd.DataFrame(input_record, columns=['Hodnoty']).to_csv(out_path('tests_single_column.csv'), index=False, header=False)

print("\n" + "=" * 86)
print("             GENERATION OF PROTOCOL AND POST-PROCESSING COMPLETED               ")
print("=" * 86)

try:
    import Protocol
    pdf_file_path = out_path(f"Protocol_{safe_mat_name}_Case_{case_number}.pdf")
    Protocol.create_Preview_Protocol(pdf_file_path, input_file_path=input_file_path, output_dir=output_dir)
    print(f"  [✓] PDF Protocol successfully generated: '{pdf_file_path}'")
except Exception as e:
    print(f"  [!] PDF Protocol warning: ({e})")

print("=" * 86 + "\n")