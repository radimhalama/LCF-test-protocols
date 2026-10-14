"""
Name:           LCF Evaluation Preview Protocol
Author:         Michal Kořínek & Radim Halama
Institution:    VSB - Technical University of Ostrava
Department:     Department of Applied Mechanics, FME
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import os
import glob
import textwrap
import re

from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# ROBUST BIBTEX PARSER (HANDLES SCOPUS NESTED BRACES & MULTI-LINE FIELDS)
# =============================================================================
def _parse_bibtex_entry(bib_file):
    """Parses any .bib file reliably, supporting nested braces like {Ti-Cu}."""
    # 1. Try bibtexparser if installed
    try:
        import bibtexparser
        with open(bib_file, 'r', encoding='utf-8') as bf:
            try:
                db = bibtexparser.load(bf)
                if db.entries:
                    return {k.lower(): v for k, v in db.entries[0].items()}
            except AttributeError:
                bf.seek(0)
                lib = bibtexparser.parse_file(bib_file)
                if lib.entries:
                    return {k.lower(): v.value for k, v in lib.entries[0].fields_dict.items()}
    except Exception:
        pass

    # 2. Built-in brace-counting parser (never fails on nested braces)
    try:
        with open(bib_file, 'r', encoding='utf-8', errors='ignore') as bf:
            content = bf.read()
        
        entry = {}
        pos = 0
        while pos < len(content):
            m = re.search(r'(\w+)\s*=\s*[\{"]', content[pos:], re.IGNORECASE)
            if not m:
                break
            key = m.group(1).lower()
            start = pos + m.end() - 1
            delim = content[start]
            if delim == '{':
                brace_count = 1
                i = start + 1
                while i < len(content) and brace_count > 0:
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                    i += 1
                val = content[start + 1:i - 1]
                pos = i
            else:  # quote "
                end = content.find('"', start + 1)
                if end == -1:
                    break
                val = content[start + 1:end]
                pos = end + 1
            
            clean_val = re.sub(r'[\{\}]', '', val).strip().replace('\n', ' ')
            clean_val = re.sub(r'\s+', ' ', clean_val)
            entry[key] = clean_val
            
        return entry
    except Exception:
        return {}


def get_lcf_citation(output_dir=None, file_path=None, direct_citation=None):
    """
    Finds citation from direct input, citation.txt, or any .bib file
    across output_dir, file_path directory, and root folder.
    """
    # 0. Check direct in-memory citation first
    if direct_citation and isinstance(direct_citation, dict) and direct_citation.get('text'):
        return direct_citation

    # 1. Search paths for citation.txt and *.bib
    search_dirs = []
    if output_dir:
        search_dirs.append(output_dir)
    if file_path and os.path.dirname(file_path):
        search_dirs.append(os.path.dirname(file_path))
    search_dirs.extend([".", ".."])
    
    seen = set()
    search_dirs = [d for d in search_dirs if d and not (d in seen or seen.add(d))]

    # Check citation.txt
    for s_dir in search_dirs:
        c_txt = os.path.join(s_dir, "citation.txt")
        if os.path.exists(c_txt):
            try:
                with open(c_txt, 'r', encoding='utf-8') as cf:
                    lines = [l.strip() for l in cf if l.strip()]
                if lines:
                    doi_url = lines[1] if len(lines) > 1 else ""
                    return {'text': lines[0], 'doi_url': doi_url}
            except Exception:
                pass

    # Check *.bib files
    for s_dir in search_dirs:
        bib_files = glob.glob(os.path.join(s_dir, "*.bib"))
        if bib_files:
            entry = _parse_bibtex_entry(bib_files[0])
            if entry:
                author = entry.get('author', '')
                if ' and ' in author:
                    authors = author.split(' and ')
                    author_str = f"{authors[0].strip()} et al." if len(authors) > 2 else ", ".join([a.strip() for a in authors])
                else:
                    author_str = author.strip()

                year = entry.get('year', '')
                title = entry.get('title', '').rstrip('.')
                if len(title) > 85:
                    title = title[:82].rstrip() + "..."

                journal = entry.get('journal', entry.get('booktitle', ''))
                vol = entry.get('volume', '')
                pages = entry.get('pages', '')
                raw_doi = entry.get('doi', entry.get('url', '')).strip()
                
                doi_url = ""
                if raw_doi:
                    if raw_doi.startswith("http"):
                        doi_url = raw_doi
                    elif "doi.org/" in raw_doi:
                        doi_url = f"https://doi.org/{raw_doi.split('doi.org/')[-1].strip()}"
                    else:
                        doi_url = f"https://doi.org/{raw_doi}"

                parts = [p for p in [author_str, f"({year})" if year else "", f'"{title}."' if title else "", journal, vol, pages] if p]
                return {'text': " ".join(parts), 'doi_url': doi_url}

    return None


def create_Preview_Protocol(File_Path="Protocol_LCF.pdf", meta=None, cfg_full=None, **kwargs):
    output_dir = kwargs.get('output_dir') or (os.path.dirname(File_Path) if os.path.dirname(File_Path) else ".")
    direct_cite = kwargs.get('citation_info') or kwargs.get('citation')
    
    # Load citation safely across all locations
    citation = get_lcf_citation(output_dir=output_dir, file_path=File_Path, direct_citation=direct_cite)
    
    if citation and citation.get('text'):
        print(f"  [✓] Protocol citation loaded: {citation['text'][:65]}...")
    else:
        print("  [i] Protocol citation: None found.")

    meta = meta or {}
    mat_meta = cfg_full.get("Material", {}) if cfg_full else {}
    geom_meta = cfg_full.get("Specimen_Default_Geometry", {}) if cfg_full else {}

    '''MAIN PARAMETERS - INTERACTIVE WITH JSON DEFAULTS'''
    def_title = meta.get("Title", "Uniaxial LCF test report")
    Title = input(f"Enter title [{def_title}]: ") or def_title

    def_spec = meta.get("Specimen_number", "ID11 solid")
    Specimen_number = input(f"Enter specimen number [{def_spec}]: ") or def_spec

    def_mat = mat_meta.get("Name", meta.get("Material", "Al5X1 EOS(SLM Z)"))
    Material = input(f"Enter material [{def_mat}]: ") or def_mat

    def_equip = meta.get("Equipment", "LabControl 100kN/120Nm")
    Equipment = input(f"Enter equipment [{def_equip}]: ") or def_equip

    def_inst = meta.get("Institution", "VSB-TUO, Czechia")
    Institution = input(f"Enter institution [{def_inst}]: ") or def_inst

    def_temp = meta.get("Temperature", "22°C")
    Temperature = input(f"Enter temperature [{def_temp}]: ") or def_temp

    def_amp = meta.get("Amplitude", "2.0 %")
    Amplitude = input(f"Enter amplitude [{def_amp}]: ") or def_amp

    def_lrate = meta.get("Loading_Rate", "1.0 %/s")
    Loading_rate = input(f"Enter loading rate [{def_lrate}]: ") or def_lrate

    def_freq = meta.get("Frequency_Hz", "0.125 Hz")
    Frequency_of_load = input(f"Enter frequency of load [{def_freq}]: ") or def_freq

    def_date = meta.get("Date", "1. 9. 2026")
    Date = input(f"Enter date [{def_date}]: ") or def_date

    def_dim = meta.get("Dimensions_text", f"{geom_meta.get('D2_inner_mm', 9.04)}/{geom_meta.get('D1_outer_mm', 10.94)} mm")
    Dimensions = input(f"Dimensions of cross-section [{def_dim}]: ") or def_dim

    def_tol = meta.get("Type_of_loading", "uniaxial")
    Type_of_loading = input(f"Enter type of loading [{def_tol}]: ") or def_tol

    def_gl = meta.get("Gauge_length_text", f"{geom_meta.get('Gauge_length_mm', 10.0)} mm")
    Gauge_length = input(f"Enter gauge length [{def_gl}]: ") or def_gl

    def_exp = meta.get("Experimenter", "Radim Halama")
    Experimenter = input(f"Enter Experimenter [{def_exp}]: ") or def_exp

    def_ctrl = meta.get("Controlled_Signal_Type", "Strain-triangular")
    Controlled = input(f"Enter controlled [{def_ctrl}]: ") or def_ctrl

    def_mean = meta.get("Mean_value", "0.0%")
    Mean_value = input(f"Enter mean value [{def_mean}]: ") or def_mean

    def_urate = meta.get("Unloading_Rate", "1.0 %/s")
    Unloading_rate = input(f"Enter unloading rate [{def_urate}]: ") or def_urate

    def_sfreq = meta.get("Sampling_Frequency", "200 Hz")
    Sampling_frequency = input(f"Enter sampling frequency [{def_sfreq}]: ") or def_sfreq

    def_ack = meta.get("Acknowledgments", "This work was supported by the Specific Research project SP2026/027.")
    Acknowledgments = input(f"Enter Acknowledgments [{def_ack}]: ") or def_ack

    def_note = meta.get("Note", "Evaluated with FABER LCF Tool.")
    Note = input(f"Enter note [{def_note}]: ") or def_note

    # Safe numeric DataFrame reading
    df_half = pd.read_csv('output_half.csv', header=0)
    data_Table = df_half.apply(pd.to_numeric, errors='coerce').values

    df_eval = pd.read_csv('Evaluation_input.csv', header=0)
    Evaluation_input = df_eval.apply(pd.to_numeric, errors='coerce').values
    
    A4_WIDTH_CM = 21.0
    A4_HEIGHT_CM = 29.7
    CM_TO_INCH = 1.0 / 2.54
        
    A4_WIDTH_IN = A4_WIDTH_CM * CM_TO_INCH
    A4_HEIGHT_IN = A4_HEIGHT_CM * CM_TO_INCH
      
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 12
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    with PdfPages(File_Path) as pdf:
        # =====================================================================
        # PAGE 1: PRIMARY TEST REPORT
        # =====================================================================
        fig = plt.figure(figsize=(A4_WIDTH_IN, A4_HEIGHT_IN), dpi=500)
        
        '''HEADER'''
        fig.text(10.5/A4_WIDTH_CM, 26.5/A4_HEIGHT_CM, 'POST-PROCESSING RECORD', ha='center', fontsize=20, weight='bold')
        try:
            logo_img = mpimg.imread("Header_logo.png")
            ax_logo = fig.add_axes([0.5/A4_WIDTH_CM, 27.5/A4_HEIGHT_CM, 20/A4_WIDTH_CM, 1.6/A4_HEIGHT_CM])
            ax_logo.imshow(logo_img)
            ax_logo.axis('off')
        except FileNotFoundError:
            fig.text(13.5/A4_WIDTH_CM + 0.15, 23/A4_HEIGHT_CM + 0.184, 'VSB - Technical University of Ostrava',
                     ha='center', va='center', fontsize=8, color='red')
        
        '''FIRST COLUMN'''
        fig.text(3/A4_WIDTH_CM, 25.5/A4_HEIGHT_CM, 'Title:')
        fig.text(7/A4_WIDTH_CM, 25.5/A4_HEIGHT_CM, Title)
        fig.text(3/A4_WIDTH_CM, 25/A4_HEIGHT_CM, 'Specimen number:')
        fig.text(7/A4_WIDTH_CM, 25/A4_HEIGHT_CM, Specimen_number)
        fig.text(3/A4_WIDTH_CM, 24.5/A4_HEIGHT_CM, 'Material:')
        fig.text(7/A4_WIDTH_CM, 24.5/A4_HEIGHT_CM, Material)
        fig.text(3/A4_WIDTH_CM, 24/A4_HEIGHT_CM, 'Equipment:')
        fig.text(7/A4_WIDTH_CM, 24/A4_HEIGHT_CM, Equipment)
        fig.text(3/A4_WIDTH_CM, 23.5/A4_HEIGHT_CM, 'Institution:')
        fig.text(7/A4_WIDTH_CM, 23.5/A4_HEIGHT_CM, Institution)
        fig.text(3/A4_WIDTH_CM, 23/A4_HEIGHT_CM, 'Temperature:')
        fig.text(7/A4_WIDTH_CM, 23/A4_HEIGHT_CM, Temperature)
        fig.text(3/A4_WIDTH_CM, 22.5/A4_HEIGHT_CM, 'Amplitude:')
        fig.text(7/A4_WIDTH_CM, 22.5/A4_HEIGHT_CM, Amplitude)
        fig.text(3/A4_WIDTH_CM, 22/A4_HEIGHT_CM, 'Loading rate:')
        fig.text(7/A4_WIDTH_CM, 22/A4_HEIGHT_CM, Loading_rate)
        fig.text(3/A4_WIDTH_CM, 21.5/A4_HEIGHT_CM, 'Frequency of load.:')
        fig.text(7/A4_WIDTH_CM, 21.5/A4_HEIGHT_CM, Frequency_of_load)
            
        '''SECOND COLUMN'''
        fig.text(12/A4_WIDTH_CM, 25.5/A4_HEIGHT_CM, 'Date:')
        fig.text(16/A4_WIDTH_CM, 25.5/A4_HEIGHT_CM, Date)
        fig.text(12/A4_WIDTH_CM, 25/A4_HEIGHT_CM, 'Dimensions:')
        fig.text(16/A4_WIDTH_CM, 25/A4_HEIGHT_CM, Dimensions)
        fig.text(12/A4_WIDTH_CM, 24.5/A4_HEIGHT_CM, 'Type of loading:')
        fig.text(16/A4_WIDTH_CM, 24.5/A4_HEIGHT_CM, Type_of_loading)
        fig.text(12/A4_WIDTH_CM, 24/A4_HEIGHT_CM, 'Gauge length:')
        fig.text(16/A4_WIDTH_CM, 24/A4_HEIGHT_CM, Gauge_length)
        fig.text(12/A4_WIDTH_CM, 23.5/A4_HEIGHT_CM, 'Experimenter:')
        fig.text(16/A4_WIDTH_CM, 23.5/A4_HEIGHT_CM, Experimenter)
        fig.text(12/A4_WIDTH_CM, 23/A4_HEIGHT_CM, 'Controlled:')
        fig.text(16/A4_WIDTH_CM, 23/A4_HEIGHT_CM, Controlled)
        fig.text(12/A4_WIDTH_CM, 22.5/A4_HEIGHT_CM, 'Mean value:')
        fig.text(16/A4_WIDTH_CM, 22.5/A4_HEIGHT_CM, Mean_value)
        fig.text(12/A4_WIDTH_CM, 22/A4_HEIGHT_CM, 'Unloading rate:')
        fig.text(16/A4_WIDTH_CM, 22/A4_HEIGHT_CM, Unloading_rate)
        fig.text(12/A4_WIDTH_CM, 21.5/A4_HEIGHT_CM, 'Sampling frequency:')
        fig.text(16/A4_WIDTH_CM, 21.5/A4_HEIGHT_CM, Sampling_frequency)
        
        ax_line = fig.add_axes([0, 0, 1, 1])
        ax_line.set_xlim(0, A4_WIDTH_CM)
        ax_line.set_ylim(0, A4_HEIGHT_CM)
        ax_line.axis('off')
        ax_line.plot([0.5, 20.5], [21, 21], color='black', linewidth=0.75)
        ax_line.plot([0.5, 20.5], [20.9, 20.9], color='black', linewidth=0.75)
        
        '''Input data for evaluation'''
        fig.text(3/A4_WIDTH_CM, 20.5/A4_HEIGHT_CM, 'Input data for evaluation')
        fig.text(3/A4_WIDTH_CM, 20/A4_HEIGHT_CM, 'N1: ' + str(int(Evaluation_input[0, 0])))
        fig.text(3/A4_WIDTH_CM, 19.5/A4_HEIGHT_CM, 'N2: ' + str(int(Evaluation_input[0, 1])))
        fig.text(6/A4_WIDTH_CM, 20/A4_HEIGHT_CM, 'Young modulus E: ' + str(int(Evaluation_input[0, 5])) + ' MPa')
        fig.text(6/A4_WIDTH_CM, 19.5/A4_HEIGHT_CM, 'E - used for Wp: ' + str(int(Evaluation_input[0, 2])) + ' MPa')
        fig.text(12/A4_WIDTH_CM, 20/A4_HEIGHT_CM, r'El. region size $\Delta\sigma_{LIN}$: ' + str(int(Evaluation_input[0, 3])) + ' MPa')
        fig.text(12/A4_WIDTH_CM, 19.5/A4_HEIGHT_CM, 'Offset pl. strain: ' + str(f"{Evaluation_input[0, 4]:.3e}"))
            
        '''PAGE 1 FIGURES'''
        fig1 = mpimg.imread("Dynamic modulus - comparison.png") if os.path.exists("Dynamic modulus - comparison.png") else np.ones((100, 100, 3))
        fig2 = mpimg.imread("HystLoopHalflife.png") if os.path.exists("HystLoopHalflife.png") else np.ones((100, 100, 3))
        fig3 = mpimg.imread("Amplitude-mean strain.png") if os.path.exists("Amplitude-mean strain.png") else np.ones((100, 100, 3))
        fig4 = mpimg.imread("Cyclic ye vs Amp Mean stress.png") if os.path.exists("Cyclic ye vs Amp Mean stress.png") else np.ones((100, 100, 3))
        imgs = [fig1, fig2, fig3, fig4]
            
        width_cm = 10.0
        height_cm = 3 * width_cm / 5.0
        width_rel = width_cm / A4_WIDTH_CM
        height_rel = height_cm / A4_HEIGHT_CM
            
        position_cm = [(0.5, 7), (0.5, 13), (10.5, 7), (10.5, 13)]
        position_rel = [[x / A4_WIDTH_CM, y / A4_HEIGHT_CM, width_rel, height_rel] for x, y in position_cm]
          
        for img, pos in zip(imgs, position_rel):
            ax = fig.add_axes(pos)
            ax.imshow(img, aspect='auto')    
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
        
        # =====================================================================
        # PAGE 1 TABLE: HALF-LIFE VALUES WITH CONDITIONAL Nf_E10 & eps_ap
        # =====================================================================
        val_eps_ap_pct = np.nan
        if 'Eps_Ap' in df_half.columns and pd.notna(df_half['Eps_Ap'].iloc[0]):
            val_eps_ap_pct = float(df_half['Eps_Ap'].iloc[0]) * 100.0
        elif os.path.exists('amp_mean_values.csv'):
            try:
                df_amp = pd.read_csv('amp_mean_values.csv')
                if 'eps_ap' in df_amp.columns and len(df_amp['eps_ap'].dropna()) > 0:
                    mid_i = len(df_amp) // 2
                    val_eps_ap_pct = float(df_amp['eps_ap'].iloc[mid_i]) * 100.0
            except Exception:
                pass
        if np.isnan(val_eps_ap_pct):
            try:
                ea = float(df_half['Eps_A'].iloc[0])
                sa = float(df_half['Sig_A'].iloc[0])
                et = float(df_half['E_dyn_tens'].iloc[0])
                ec = float(df_half['E_dyn_comp'].iloc[0])
                val_eps_ap_pct = abs(ea - 2.0 * sa / (et + ec)) * 100.0
            except Exception:
                val_eps_ap_pct = 0.0

        val_nf_e10 = None
        if 'Nf_E10%' in df_half.columns:
            raw_nf10 = df_half['Nf_E10%'].iloc[0]
            if pd.notna(raw_nf10) and str(raw_nf10).strip() not in ['', 'None', 'nan', 'NaN', '-1']:
                try:
                    val_nf_e10 = int(float(raw_nf10))
                except Exception:
                    pass

        headers_p1 = [
            r'$\epsilon_{a}$ (%)',
            r'$\epsilon_{ap}$ (%)',
            r'$\sigma_{a}$ (MPa)',
            r'$\sigma_{m}$ (MPa)',
            r'$N_{f}$ (-)'
        ]
        values_p1 = [
            np.round(float(df_half['Eps_A'].iloc[0]) * 100.0, 3),
            np.round(val_eps_ap_pct, 3),
            np.round(float(df_half['Sig_A'].iloc[0]), 2),
            np.round(float(df_half['Sig_M'].iloc[0]), 2),
            int(df_half['Nf'].iloc[0])
        ]

        if val_nf_e10 is not None:
            headers_p1.append(r'$N_{f}^{E10}$ (-)')
            values_p1.append(val_nf_e10)

        headers_p1.extend([r'$E_{dyn}^{upp}$ (MPa)', r'$E_{dyn}^{btm}$ (MPa)', r'$Y_{cyclic}$ (MPa)'])
        values_p1.extend([
            np.round(float(df_half['E_dyn_tens'].iloc[0]), 2),
            np.round(float(df_half['E_dyn_comp'].iloc[0]), 2),
            np.round(float(df_half['Y_cyclic'].iloc[0]), 2)
        ])

        data_tab = [headers_p1, values_p1]
            
        num_cols = len(data_tab[0])
        total_tab_w_cm = 18.0
        cell_width_cm = total_tab_w_cm / num_cols
        cell_height_cm = 0.75
        x_center_cm = 10.5
        x_tab = x_center_cm - (cell_width_cm * num_cols) / 2
        y_tab = 6.0
            
        ax_tab = fig.add_axes([0, 0, 1, 1])
        ax_tab.set_xlim(0, A4_WIDTH_CM)
        ax_tab.set_ylim(0, A4_HEIGHT_CM)
        ax_tab.axis('off')
            
        fig.text(x_tab / A4_WIDTH_CM, (y_tab + cell_height_cm / 4) / A4_HEIGHT_CM, 'Half-life values:')
            
        for row_index, row in enumerate(data_tab):
            for col_index, cell in enumerate(row):
                x = x_tab + col_index * cell_width_cm
                y = y_tab - row_index * cell_height_cm
                rect = plt.Rectangle((x, y - cell_height_cm), cell_width_cm, cell_height_cm,
                                     fill=False, edgecolor='black', linewidth=0.75)
                ax_tab.add_patch(rect)
                font_sz = 10 if num_cols >= 9 else 11
                ax_tab.text(x + cell_width_cm / 2, y - cell_height_cm / 2, str(cell),
                            ha='center', va='center', fontsize=font_sz)
            
        # =====================================================================
        # PAGE 1 FOOTER: REORGANIZED WITHOUT OVERLAPS
        # =====================================================================
        URL = "https://faber-cost.eu/"
        wrapped_text = textwrap.fill("Note: " + Note, width=110)
        fig.text(0.5 / A4_WIDTH_CM, 4.0 / A4_HEIGHT_CM, wrapped_text, fontsize=12, ha='left', va='top')

        fig.text(10.5 / A4_WIDTH_CM, 2.3 / A4_HEIGHT_CM, "Acknowledgments: " + Acknowledgments,
                 fontsize=9.5, ha='center', va='center', fontstyle='italic')
        fig.text(10.5 / A4_WIDTH_CM, 1.7 / A4_HEIGHT_CM, "Copyright © 2026 FABER Cost Action 23109, All rights reserved.",
                 fontsize=9.5, ha='center', va='center', url=URL, fontstyle='italic')

        # Citation placed in dedicated lower footer space
        if citation and citation.get('text'):
            fig.text(0.5 / A4_WIDTH_CM, 1.05 / A4_HEIGHT_CM, f"Ref: {citation['text']}",
                     fontsize=7.5, ha='left', va='center', fontstyle='italic', color='#222222')
            if citation.get('doi_url'):
                fig.text(0.5 / A4_WIDTH_CM, 0.55 / A4_HEIGHT_CM, "DOI: ",
                         fontsize=7.5, ha='left', va='center', color='#222222')
                fig.text((0.5 + 0.68) / A4_WIDTH_CM, 0.55 / A4_HEIGHT_CM, citation['doi_url'],
                         fontsize=7.5, ha='left', va='center', color='#0044cc',
                         url=citation['doi_url'])

        fig.text(20.0 / A4_WIDTH_CM, 0.55 / A4_HEIGHT_CM, "Page 1/2", fontsize=9.5, ha='right', va='center')

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # PAGE 2: 6 SUPPLEMENTARY FIGURES + ADDITIONAL VALUES TABLE
        # =====================================================================
        fig_p2 = plt.figure(figsize=(A4_WIDTH_IN, A4_HEIGHT_IN), dpi=300)

        '''PAGE 2 FIGURES'''
        page2_images = [
            "Young's Modulus.png",                     # Row 1, Left
            "stress_strain_time_history.png",          # Row 1, Right
            "E_compression_Nf10_determination.png",     # Row 2, Left
            "initial_hysteresis_loops_10_cycles.png",   # Row 2, Right
            "Nf.png",                                  # Row 3, Left
            "WP_vs_N.png",                             # Row 3, Right
        ]

        width_cm_p2 = 10.0
        height_cm_p2 = 3.0 * width_cm_p2 / 5.0
        width_rel_p2 = width_cm_p2 / A4_WIDTH_CM
        height_rel_p2 = height_cm_p2 / A4_HEIGHT_CM

        position_cm_p2 = [
            (0.5, 22.5), (10.5, 22.5),
            (0.5, 15.5), (10.5, 15.5),
            (0.5, 8.5),  (10.5, 8.5),
        ]

        for img_name, (x_pos, y_pos) in zip(page2_images, position_cm_p2):
            ax = fig_p2.add_axes([x_pos / A4_WIDTH_CM, y_pos / A4_HEIGHT_CM, width_rel_p2, height_rel_p2])
            try:
                img_data = mpimg.imread(img_name)
                ax.imshow(img_data, aspect='auto')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
            except FileNotFoundError:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.5, 0.5, f"Placeholder:\n'{img_name}'\n(File not found)",
                        ha='center', va='center', fontsize=9, color='gray')
                for spine in ax.spines.values():
                    spine.set_linestyle('--')
                    spine.set_color('gray')
                    spine.set_linewidth(1.0)

        # 1. W_p^tot
        col_fat = 'Wp_tot' if 'Wp_tot' in df_half.columns else 'Wp_tot (MPa)'
        val_wp_fat = float(df_half[col_fat].iloc[0]) if col_fat in df_half.columns and pd.notna(df_half[col_fat].iloc[0]) else np.nan
        val_wp_fat_str = f"{val_wp_fat:.2f}" if not np.isnan(val_wp_fat) else "NaN"

        # 2. Delta W_p^mean
        col_mean = 'Wp_mean' if 'Wp_mean' in df_half.columns else 'Wp_mean (MPa)'
        val_wp_mean = float(df_half[col_mean].iloc[0]) if col_mean in df_half.columns and pd.notna(df_half[col_mean].iloc[0]) else np.nan
        val_wp_mean_str = f"{val_wp_mean:.3f}" if not np.isnan(val_wp_mean) else "NaN"

        # 3. Delta W_p
        col_dwp = 'DeltaWp' if 'DeltaWp' in df_half.columns else 'DeltaWp (MPa)'
        val_dwp = float(df_half[col_dwp].iloc[0]) if col_dwp in df_half.columns and pd.notna(df_half[col_dwp].iloc[0]) else np.nan
        val_dwp_str = f"{val_dwp:.3f}" if not np.isnan(val_dwp) else "NaN"

        # 4. Delta W_p^VarE
        col_vare = 'DeltaWp_VarE' if 'DeltaWp_VarE' in df_half.columns else 'DeltaWp_VarE (MPa)'
        val_wp_vare = float(df_half[col_vare].iloc[0]) if col_vare in df_half.columns and pd.notna(df_half[col_vare].iloc[0]) else val_dwp
        val_wp_vare_str = f"{val_wp_vare:.3f}" if not np.isnan(val_wp_vare) else "NaN"

        # PAGE 2 TABLE
        data_tab_p2 = [
            [r'$W_{p}^{tot}$ (MPa)', r'$\Delta W_{p}^{mean}$ (MPa)', r'$\Delta W_{p}$ (MPa)', r'$\Delta W_{p}^{VarE}$ (MPa)'],
            [val_wp_fat_str, val_wp_mean_str, val_dwp_str, val_wp_vare_str]
        ]

        cell_w_p2 = 3.5
        cell_h_p2 = 0.75
        num_cols_p2 = len(data_tab_p2[0])
        x_center_cm = 10.5
        x_tab_p2 = x_center_cm - (cell_w_p2 * num_cols_p2) / 2
        y_tab_p2 = 6.2

        ax_tab_p2 = fig_p2.add_axes([0, 0, 1, 1])
        ax_tab_p2.set_xlim(0, A4_WIDTH_CM)
        ax_tab_p2.set_ylim(0, A4_HEIGHT_CM)
        ax_tab_p2.axis('off')

        fig_p2.text(x_tab_p2 / A4_WIDTH_CM, (y_tab_p2 + cell_h_p2 / 4) / A4_HEIGHT_CM,
                    'Additional values:', fontsize=12, weight='bold')

        for r_idx, row in enumerate(data_tab_p2):
            for c_idx, cell in enumerate(row):
                x_c = x_tab_p2 + c_idx * cell_w_p2
                y_c = y_tab_p2 - r_idx * cell_h_p2
                rect = plt.Rectangle((x_c, y_c - cell_h_p2), cell_w_p2, cell_h_p2,
                                     fill=False, edgecolor='black', linewidth=0.75)
                ax_tab_p2.add_patch(rect)
                ax_tab_p2.text(x_c + cell_w_p2 / 2, y_c - cell_h_p2 / 2, str(cell),
                               ha='center', va='center', fontsize=12)

        # =====================================================================
        # PAGE 2 FOOTER: CITATION & NUMBERING (NO OVERLAPS)
        # =====================================================================
        if citation and citation.get('text'):
            fig_p2.text(0.5 / A4_WIDTH_CM, 1.05 / A4_HEIGHT_CM, f"Ref: {citation['text']}",
                        fontsize=7.5, ha='left', va='center', fontstyle='italic', color='#222222')
            if citation.get('doi_url'):
                fig_p2.text(0.5 / A4_WIDTH_CM, 0.55 / A4_HEIGHT_CM, "DOI: ",
                            fontsize=7.5, ha='left', va='center', color='#222222')
                fig_p2.text((0.5 + 0.68) / A4_WIDTH_CM, 0.55 / A4_HEIGHT_CM, citation['doi_url'],
                            fontsize=7.5, ha='left', va='center', color='#0044cc',
                            url=citation['doi_url'])
        
        fig_p2.text(20.0 / A4_WIDTH_CM, 0.55 / A4_HEIGHT_CM, "Page 2/2", fontsize=9.5, ha='right', va='center')

        pdf.savefig(fig_p2)
        plt.close(fig_p2)

    try:
        os.startfile(File_Path)
    except Exception:
        pass
    
    return {
        "Title": Title, "Specimen_number": Specimen_number, "Material": Material,
        "Equipment": Equipment, "Institution": Institution, "Temperature": Temperature,
        "Amplitude": Amplitude, "Loading_Rate": Loading_rate, "Frequency_Hz": Frequency_of_load,
        "Date": Date, "Dimensions_text": Dimensions, "Type_of_loading": Type_of_loading,
        "Gauge_length_text": Gauge_length, "Experimenter": Experimenter, "Controlled_Signal_Type": Controlled,
        "Mean_value": Mean_value, "Unloading_Rate": Unloading_rate, "Sampling_Frequency": Sampling_frequency,
        "Acknowledgments": Acknowledgments, "Note": Note
    }