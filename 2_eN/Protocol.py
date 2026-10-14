"""
Name:           E-N CURVE PROTOCOL UNIAXIAL (2-PAGE STANDARDIZED LAYOUT)
Author:         Michal Korinek & Radim Halama
Institution:    VSB - Technical University of Ostrava
Department:     Department of Applied Mechanics
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import os
import sys
import glob
import re

from matplotlib.backends.backend_pdf import PdfPages

PLOT_W_CM = 10.0
PLOT_H_CM = 6.5

PAGE_W_CM = 21.0
PAGE_H_CM = 29.7
CM_TO_INCH = 1.0 / 2.54


# =============================================================================
# HELPER 1: BIBTEX CITATION PARSER (SCOPUS / LATEX .BIB / MANUAL TXT)
# =============================================================================
def get_bib_citation(bib_path=None, output_dir=None):
    """
    Parses a .bib file, or reads citation.txt from output_dir if available.
    Returns a dictionary with formatted text, clean DOI, and clickable URL.
    """
    # 1. Check for manual citation file saved in output folder
    if output_dir:
        cite_file = os.path.join(output_dir, 'citation.txt')
        if os.path.exists(cite_file):
            try:
                with open(cite_file, 'r', encoding='utf-8') as cf:
                    lines = [l.strip() for l in cf if l.strip()]
                if lines:
                    ref_text = lines[0]
                    doi_url = lines[1] if len(lines) > 1 else ""
                    doi_clean = doi_url.split("doi.org/")[-1].strip() if "doi.org/" in doi_url else doi_url
                    return {'text': ref_text, 'doi': doi_clean, 'doi_url': doi_url}
            except Exception:
                pass

    # 2. Check for .bib file
    if not bib_path:
        bib_files = glob.glob("*.bib")
        if not bib_files:
            return None
        bib_path = bib_files[0]
        
    if not os.path.exists(bib_path):
        return None

    entry = {}
    try:
        import bibtexparser
        with open(bib_path, 'r', encoding='utf-8') as bf:
            try:
                bib_db = bibtexparser.load(bf)
                if bib_db.entries:
                    entry = bib_db.entries[0]
            except AttributeError:
                bf.seek(0)
                library = bibtexparser.parse_file(bib_path)
                if library.entries:
                    entry = {k.lower(): v.value for k, v in library.entries[0].fields_dict.items()}
    except Exception:
        pass

    if not entry:
        try:
            with open(bib_path, 'r', encoding='utf-8', errors='ignore') as bf:
                content = bf.read()
            fields = re.findall(r'(\w+)\s*=\s*[\{"]([^\{\}"]+)[\}"]', content, re.IGNORECASE)
            for k, v in fields:
                entry[k.lower()] = v.strip().replace('\n', ' ')
        except Exception:
            return None

    if not entry:
        return None

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
    doi_clean = ""
    doi_url = ""
    if raw_doi:
        if "doi.org/" in raw_doi:
            doi_clean = raw_doi.split("doi.org/")[-1].strip()
            doi_url = raw_doi if raw_doi.startswith("http") else f"https://{raw_doi}"
        else:
            doi_clean = raw_doi
            doi_url = f"https://doi.org/{raw_doi}"

    parts = []
    if author_str:
        parts.append(f"{author_str}")
    if year:
        parts.append(f"({year})")
    if title:
        parts.append(f"\"{title}.\"")
    if journal:
        j_str = journal
        if volume:
            j_str += f" {volume}"
        if pages:
            j_str += f": {pages}"
        parts.append(j_str)

    return {
        'text': " ".join(parts),
        'doi': doi_clean,
        'doi_url': doi_url
    }


# =============================================================================
# HELPER 2: NON-STRETCHING ASPECT-PRESERVING IMAGE PLACEMENT
# =============================================================================
def draw_image_box(fig, img_path, box_x_cm, box_y_cm, box_w_cm, box_h_cm):
    """
    Fits an image maintaining its native aspect ratio.
    Returns (ax, (x_pos, y_pos, w_fit, h_fit)) in centimeters.
    """
    if not os.path.exists(img_path):
        return None, (box_x_cm, box_y_cm, box_w_cm, box_h_cm)
    
    img = mpimg.imread(img_path)
    h_px, w_px = img.shape[0], img.shape[1]
    img_aspect = w_px / h_px
    box_aspect = box_w_cm / box_h_cm

    if img_aspect > box_aspect:
        w_fit = box_w_cm
        h_fit = box_w_cm / img_aspect
    else:
        h_fit = box_h_cm
        w_fit = box_h_cm * img_aspect

    x_pos = box_x_cm + (box_w_cm - w_fit) / 2.0
    y_pos = box_y_cm + (box_h_cm - h_fit) / 2.0

    ax = fig.add_axes([x_pos / PAGE_W_CM, y_pos / PAGE_H_CM, w_fit / PAGE_W_CM, h_fit / PAGE_H_CM])
    ax.imshow(img, aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    return ax, (x_pos, y_pos, w_fit, h_fit)


# =============================================================================
# HELPER 3: DYNAMIC-WIDTH TABLE PLACEMENT
# =============================================================================
def draw_table(fig, data, x_cm, y_top_cm, col_widths_cm, row_h_cm=0.50, font_size=9.0):
    """
    Draws a table with per-column widths and returns the bottom Y coordinate.
    """
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W_CM)
    ax.set_ylim(0, PAGE_H_CM)
    ax.axis('off')

    for r_idx, row in enumerate(data):
        curr_x = x_cm
        y_cell = y_top_cm - (r_idx + 1) * row_h_cm
        for c_idx, cell in enumerate(row):
            w_cell = col_widths_cm[c_idx]
            rect = plt.Rectangle((curr_x, y_cell), w_cell, row_h_cm,
                                 fill=False, edgecolor='black', linewidth=0.75)
            ax.add_patch(rect)
            weight = 'bold' if r_idx == 0 else 'normal'
            ax.text(curr_x + w_cell / 2.0, y_cell + row_h_cm / 2.0, str(cell),
                    ha='center', va='center', fontsize=font_size, weight=weight)
            curr_x += w_cell

    y_bottom_cm = y_top_cm - len(data) * row_h_cm
    return y_bottom_cm


def format_w(val):
    """Returns '1' instead of '1.0' if decimal fraction is zero."""
    if abs(val - round(val)) < 1e-4:
        return f"{int(round(val))}"
    return f"{val:.2f}"


# =============================================================================
# HELPER 4: REUSABLE HEADER DRAWER (FOR PAGE 1 & PAGE 2)
# =============================================================================
def draw_page_header(fig, page_title, meta_author, meta_post, meta_type, meta_notes, meta_date_exp, meta_date_post):
    """Draws uniform header, logo, metadata and divider lines on any page."""
    fig.text(10.5 / PAGE_W_CM, 26.8 / PAGE_H_CM, page_title,
             ha='center', fontsize=18, weight='bold')
    
    if os.path.exists("Header_logo.png"):
        logo_img = mpimg.imread("Header_logo.png")
        ax_logo = fig.add_axes([0.5 / PAGE_W_CM, 27.7 / PAGE_H_CM, 20.0 / PAGE_W_CM, 1.3 / PAGE_H_CM])
        ax_logo.imshow(logo_img, aspect='equal')
        ax_logo.axis('off')

    # Metadata text blocks
    fig.text(1.5 / PAGE_W_CM, 25.8 / PAGE_H_CM, 'Experimenter:', weight='bold')
    fig.text(5.5 / PAGE_W_CM, 25.8 / PAGE_H_CM, meta_author)  
    fig.text(1.5 / PAGE_W_CM, 25.3 / PAGE_H_CM, 'Post-processing:', weight='bold')
    fig.text(5.5 / PAGE_W_CM, 25.3 / PAGE_H_CM, meta_post)   
    fig.text(1.5 / PAGE_W_CM, 24.8 / PAGE_H_CM, 'Loading type:', weight='bold')
    fig.text(5.5 / PAGE_W_CM, 24.8 / PAGE_H_CM, meta_type)  
    fig.text(1.5 / PAGE_W_CM, 24.3 / PAGE_H_CM, 'Notes:', weight='bold')
    fig.text(5.5 / PAGE_W_CM, 24.3 / PAGE_H_CM, meta_notes)  
    
    fig.text(15.0 / PAGE_W_CM, 25.8 / PAGE_H_CM, 'Date:', weight='bold')
    fig.text(17.0 / PAGE_W_CM, 25.8 / PAGE_H_CM, meta_date_exp)  
    fig.text(15.0 / PAGE_W_CM, 25.3 / PAGE_H_CM, 'Date:', weight='bold')
    fig.text(17.0 / PAGE_W_CM, 25.3 / PAGE_H_CM, meta_date_post) 

    # Divider lines
    ax_lines = fig.add_axes([0, 0, 1, 1])
    ax_lines.set_xlim(0, PAGE_W_CM); ax_lines.set_ylim(0, PAGE_H_CM); ax_lines.axis('off')
    ax_lines.plot([14.5, 19.5], [25.7, 25.7], color='black', linewidth=0.75)
    ax_lines.plot([14.5, 19.5], [25.2, 25.2], color='black', linewidth=0.75)
    ax_lines.plot([0.5, 20.5], [23.8, 23.8], color='black', linewidth=0.75)
    ax_lines.plot([0.5, 20.5], [23.7, 23.7], color='black', linewidth=0.75)


# =============================================================================
# MAIN PROTOCOL GENERATOR
# =============================================================================
def create_Preview_Protocol(File_Path="Protocol.pdf", input_file_path=None, bib_file_path=None, output_dir=None, citation_info=None, acknowledgments=None, **kwargs):
    if output_dir is None:
        output_dir = os.path.dirname(File_Path) if os.path.dirname(File_Path) else "."

    def p(fname):
        """Finds file inside the output directory, falling back to root directory."""
        path = os.path.join(output_dir, fname)
        return path if os.path.exists(path) else fname

    if not os.path.exists(p('tests_single_column.csv')):
        raise FileNotFoundError(f"File '{p('tests_single_column.csv')}' was not found.")
        
    df_meta = pd.read_csv(p('tests_single_column.csv'), header=None)
    metadata_loaded = df_meta[0].tolist()
    
    experimenter = str(metadata_loaded[0])
    postprocessing = str(metadata_loaded[1])
    loading_type = str(metadata_loaded[2])
    date_experiment = str(metadata_loaded[4])
    date_postprocessing = str(metadata_loaded[5])
    notes = str(metadata_loaded[6])
    
    # Transition lives
    transition_life_3D = float(metadata_loaded[7])
    
    # 9-row format input reading
    input_file = input_file_path if (input_file_path and os.path.exists(input_file_path)) else 'Inputs.txt'
    if not os.path.exists(input_file):
        matches = glob.glob('Inputs*.txt')
        if matches: 
            input_file = matches[0]

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    def parse_f(s):
        return [float(x.strip()) for x in s.replace(';', ',').split(',') if x.strip()]

    stress = parse_f(lines[0])
    strain = parse_f(lines[1])
    nf = [int(float(x)) for x in parse_f(lines[2])]
    weights = parse_f(lines[3])
    E_const = float(lines[4].strip())
    e_var = parse_f(lines[5])
    nf_type = lines[8].strip() if len(lines) > 8 else "Nf_E10"

    # Conventional 2D transition life determination
    if len(metadata_loaded) > 10 and str(metadata_loaded[10]).strip() not in ['', 'None', 'nan']:
        transition_life_2D = float(metadata_loaded[10])
    else:
        try:
            df_konv_tmp = pd.read_csv(p('mechprop_conv_const.csv'), header=None)[0].tolist()
            b_k, c_k, Sf_k, Ef_k = df_konv_tmp[2], df_konv_tmp[1], df_konv_tmp[4], df_konv_tmp[5]
            transition_life_2D = ((E_const * Ef_k / Sf_k)**(1.0 / (b_k - c_k))) / 2.0
        except Exception:
            transition_life_2D = transition_life_3D
            
    # Load bibliographic reference (use direct in-memory citation_info if provided)
    if citation_info:
        bib_citation = citation_info
    else:
        bib_citation = get_bib_citation(bib_file_path, output_dir=output_dir)

    # Acknowledgments text (default or passed)
    url_faber = "https://faber-cost.eu/"
    ack_text = acknowledgments or kwargs.get("acknowledgments", "Financial support from the Czech Science Foundation / FABER COST Action 23109.")

    A4_WIDTH_IN = PAGE_W_CM * CM_TO_INCH
    A4_HEIGHT_IN = PAGE_H_CM * CM_TO_INCH
      
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 11
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    with PdfPages(File_Path) as pdf:
        # =====================================================================
        # PAGE 1: PRIMARY TEST REPORT
        # =====================================================================
        fig1 = plt.figure(figsize=(A4_WIDTH_IN, A4_HEIGHT_IN), dpi=500)
        
        # 1. Header on Page 1
        draw_page_header(fig1, 'E-N CURVE PROTOCOL - PUSH-PULL (Page 1)',
                         experimenter, postprocessing, loading_type, notes, date_experiment, date_postprocessing)

        # 2. Page 1 Figures
        # Top-Right: CSSC
        _, (_, cssc_y, _, cssc_h) = draw_image_box(
            fig1, p("xy_plane.png"), box_x_cm=10.5, box_y_cm=16.5, box_w_cm=PLOT_W_CM, box_h_cm=PLOT_H_CM
        )
        plot_top_p1 = cssc_y + cssc_h

        # Bottom plots: 3D comparison and CMB
        img_3d_file = p("3D_comparison_Econst_vs_Evar.png") if os.path.exists(p("3D_comparison_Econst_vs_Evar.png")) else p("3D.png")
        draw_image_box(fig1, img_3d_file, box_x_cm=0.5, box_y_cm=8.8, box_w_cm=10.0, box_h_cm=7.5)
        draw_image_box(fig1, p("xz_plane.png"), box_x_cm=10.5, box_y_cm=8.8, box_w_cm=PLOT_W_CM, box_h_cm=PLOT_H_CM)
        
        # 3. Table 1: Input Data (Title states E_const value, Table displays individual E_var)
        data_input_table = [[r'ID', r'$\varepsilon_a$ [%]', r'$\sigma_a$ [MPa]', r'$N_f$ [-]', r'$E_{\mathrm{var}}$ [MPa]', 'w']]
        for i in range(len(stress)):
            var_val = e_var[i] if i < len(e_var) else E_const
            data_input_table.append([
                str(i + 1),
                f"{strain[i] * 100:.2f}",
                f"{stress[i]:.1f}",
                f"{int(nf[i])}",
                f"{var_val:.0f}",
                format_w(weights[i])
            ])
            
        col_w_tab1 = [1.0, 1.8, 2.0, 1.6, 2.3, 1.0] # Total: 9.7 cm
        
        title_y_p1 = plot_top_p1
        table_top_p1 = title_y_p1 - 0.45
        
        fig1.text(0.5 / PAGE_W_CM, title_y_p1 / PAGE_H_CM, 
                  rf'Input Data (Evaluated with $E_{{\mathrm{{const}}}} = {E_const:.0f}$ MPa):', 
                  fontsize=10.5, weight='bold')
        
        y_bot_t1 = draw_table(fig1, data_input_table, x_cm=0.5, y_top_cm=table_top_p1,
                              col_widths_cm=col_w_tab1, row_h_cm=0.48, font_size=8.5)
        
        if "E10" in nf_type.upper():
            criterion_desc = r'$N_f$ is defined as a 10% decrease in dynamic modulus ($N_f^{E10}$)'
        elif "25" in nf_type:
            criterion_desc = r'$N_f$ is defined as a 25% drop in stress amplitude ($N_f^{25}$)'
        else:
            criterion_desc = f'$N_f$ criterion: {nf_type}'
            
        fig1.text(0.5 / PAGE_W_CM, (y_bot_t1 - 0.40) / PAGE_H_CM, criterion_desc, fontsize=8.5, fontstyle='italic')
        
        # 4. Table 2: Material Parameters (Under 3D plot)
        df_3D_const = pd.read_csv(p('mechprop_3D_const.csv'), header=None)[0].tolist() if os.path.exists(p('mechprop_3D_const.csv')) else [0]*6
        df_3D_var   = pd.read_csv(p('mechprop_3D_var.csv'), header=None)[0].tolist() if os.path.exists(p('mechprop_3D_var.csv')) else [0]*6
        df_2D_const = pd.read_csv(p('mechprop_conv_const.csv'), header=None)[0].tolist() if os.path.exists(p('mechprop_conv_const.csv')) else [0]*6
        df_2D_var   = pd.read_csv(p('mechprop_conv_var.csv'), header=None)[0].tolist() if os.path.exists(p('mechprop_conv_var.csv')) else [0]*6
        
        data_tab2 = [
            ["Parameter", "3D (const / var)", "Conv. (const / var)"],
            [r"$K'$ [MPa]", f"{df_3D_const[3]:.1f} / {df_3D_var[3]:.1f}", f"{df_2D_const[3]:.1f} / {df_2D_var[3]:.1f}"],
            [r"$n'$ [-]", f"{df_3D_const[0]:.4f} / {df_3D_var[0]:.4f}", f"{df_2D_const[0]:.4f} / {df_2D_var[0]:.4f}"],
            [r"$\sigma'_f$ [MPa]", f"{df_3D_const[4]:.1f} / {df_3D_var[4]:.1f}", f"{df_2D_const[4]:.1f} / {df_2D_var[4]:.1f}"],
            [r"$b$ [-]", f"{df_3D_const[2]:.4f} / {df_3D_var[2]:.4f}", f"{df_2D_const[2]:.4f} / {df_2D_var[2]:.4f}"],
            [r"$\varepsilon'_f$ [-]", f"{df_3D_const[5]:.4f} / {df_3D_var[5]:.4f}", f"{df_2D_const[5]:.4f} / {df_2D_var[5]:.4f}"],
            [r"$c$ [-]", f"{df_3D_const[1]:.4f} / {df_3D_var[1]:.4f}", f"{df_2D_const[1]:.4f} / {df_2D_var[1]:.4f}"],
        ]
        
        col_w_tab2 = [2.8, 3.5, 3.5]
        fig1.text(0.5 / PAGE_W_CM, 7.7 / PAGE_H_CM, r'Material Parameters ($E_{\mathrm{const}}$ / $E_{\mathrm{var}}$):',
                  fontsize=10.0, weight='bold')
        
        y_bot_t2 = draw_table(fig1, data_tab2, x_cm=0.5, y_top_cm=7.4,
                              col_widths_cm=col_w_tab2, row_h_cm=0.48, font_size=8.5)
        
        fig1.text(0.5 / PAGE_W_CM, (y_bot_t2 - 0.40) / PAGE_H_CM,
                  f'Transition life Nt: 3D = {transition_life_3D:.0f} cycles | Conv. = {transition_life_2D:.0f} cycles',
                  fontsize=9.5, weight='bold')

        # 5. Table 3: Reliability (Under CMB plot)
        df_Rel = pd.read_csv(p('Reliability.csv'), sep=';', header=0).values if os.path.exists(p('Reliability.csv')) else np.zeros((1, 12))
        data_tab3 = [
            [r"$R^2$", "3D (const / var)", "Conv (const / var)"],
            [r"$CSSC$", f"{df_Rel[0,0]:.3f} / {df_Rel[0,3]:.3f}", f"{df_Rel[0,6]:.3f} / {df_Rel[0,9]:.3f}"],
            [r"$Basquin$", f"{df_Rel[0,1]:.3f} / {df_Rel[0,4]:.3f}", f"{df_Rel[0,7]:.3f} / {df_Rel[0,10]:.3f}"],
            [r"$Cof.-Man.$", f"{df_Rel[0,2]:.3f} / {df_Rel[0,5]:.3f}", f"{df_Rel[0,8]:.3f} / {df_Rel[0,11]:.3f}"],
        ]
        
        col_w_tab3 = [2.8, 3.5, 3.5]
        fig1.text(10.5 / PAGE_W_CM, 7.7 / PAGE_H_CM, r'Reliability $R^2$ ($E_{\mathrm{const}}$ / $E_{\mathrm{var}}$):',
                  fontsize=10.0, weight='bold')
        
        draw_table(fig1, data_tab3, x_cm=10.5, y_top_cm=7.4,
                   col_widths_cm=col_w_tab3, row_h_cm=0.48, font_size=8.5)
        
        # =====================================================================
        # PAGE 1 FOOTER: ACKNOWLEDGMENTS (ONLY ON PAGE 1) + CITATION + PAGE NO
        # =====================================================================
        fig1.text(10.5 / PAGE_W_CM, 2.1 / PAGE_H_CM, f"Acknowledgments: {ack_text}",
                  fontsize=9.0, ha='center', va='center', fontstyle='italic')
        fig1.text(10.5 / PAGE_W_CM, 1.6 / PAGE_H_CM, "Copyright © 2026 FABER Cost Action 23109, All rights reserved.",
                  fontsize=9.0, ha='center', va='center', url=url_faber, fontstyle='italic')

        if bib_citation:
            fig1.text(0.5 / PAGE_W_CM, 1.10 / PAGE_H_CM, f"Ref: {bib_citation['text']}",
                      fontsize=7.5, ha='left', va='center', fontstyle='italic', color='#222222')
            if bib_citation.get('doi_url'):
                fig1.text(0.5 / PAGE_W_CM, 0.65 / PAGE_H_CM, "DOI: ",
                          fontsize=7.5, ha='left', va='center', color='#222222')
                fig1.text((0.5 + 0.68) / PAGE_W_CM, 0.65 / PAGE_H_CM, bib_citation['doi_url'],
                          fontsize=7.5, ha='left', va='center', color='#0044cc',
                          url=bib_citation['doi_url'])
                          
        fig1.text(20.5 / PAGE_W_CM, 0.65 / PAGE_H_CM, "Page 1/2", fontsize=9.5, ha='right', va='center')
        
        pdf.savefig(fig1)
        plt.close(fig1)

        # =====================================================================
        # PAGE 2: ENERGY & COMPARISONS REPORT (CLEAN, NO ACKNOWLEDGMENTS)
        # =====================================================================
        fig2 = plt.figure(figsize=(A4_WIDTH_IN, A4_HEIGHT_IN), dpi=500)
        
        # 1. Header on Page 2
        draw_page_header(fig2, 'E-N CURVE PROTOCOL - ENERGY & COMPARISONS (Page 2)',
                         experimenter, postprocessing, loading_type, notes, date_experiment, date_postprocessing)

        # 2. Top Row Plots (Y = 16.5 cm to 23.0 cm)
        draw_image_box(fig2, p("page2_cssc_var.png"), box_x_cm=0.5, box_y_cm=16.5, box_w_cm=PLOT_W_CM, box_h_cm=PLOT_H_CM)
        draw_image_box(fig2, p("page2_cm_comparison.png"), box_x_cm=10.5, box_y_cm=16.5, box_w_cm=PLOT_W_CM, box_h_cm=PLOT_H_CM)

        # 3. Middle Row (Y = 8.8 cm to 15.3 cm)
        _, (_, energy_y, _, energy_h) = draw_image_box(
            fig2, p("page2_energy_plots.png"), box_x_cm=10.5, box_y_cm=8.8, box_w_cm=PLOT_W_CM, box_h_cm=PLOT_H_CM
        )
        plot_top_p2 = energy_y + energy_h

        if os.path.exists(p('Page2_Input_Table.csv')):
            df_p2 = pd.read_csv(p('Page2_Input_Table.csv'), sep=';')
            p2_table = [["ID", r"$\sigma_m$ [MPa]", r"$W_p$ [MPa]", r"$\mathrm{SWT}$ [MPa]"]]
            for _, row in df_p2.iterrows():
                p2_table.append([
                    str(int(row['ID'])),
                    f"{row['Sigma_m']:.1f}",
                    f"{row['W_p']:.5f}",
                    f"{row['SWT']:.5f}"
                ])
            
            col_w_p2 = [1.2, 2.6, 2.9, 2.9]
            title_y_p2 = plot_top_p2
            table_top_p2 = title_y_p2 - 0.45
            
            fig2.text(0.5 / PAGE_W_CM, title_y_p2 / PAGE_H_CM, 'Specimen Energy & Damage Parameters:', fontsize=10.5, weight='bold')
            draw_table(fig2, p2_table, x_cm=0.5, y_top_cm=table_top_p2,
                       col_widths_cm=col_w_p2, row_h_cm=0.48, font_size=8.5)

        # 4. Bottom Table (4-Parametric Fatigue Model Parameters)
        if os.path.exists(p('Fatigue_Params_4P.csv')):
            df_4p = pd.read_csv(p('Fatigue_Params_4P.csv'), sep=';')
            p4_table = [["Fatigue Parameter", "$C_1$", "$C_2$", "$C_3$", "$C_4$", r"$R^2$"]]
            for _, row in df_4p.iterrows():
                param_label = r"$\mathrm{SWT}$ [MPa]" if str(row['Parameter']).upper() == 'SWT' else r"$W_p$ [MPa]"
                p4_table.append([
                    param_label,
                    f"{row['C1']:.4e}",
                    f"{row['C2']:.4f}",
                    f"{row['C3']:.4e}",
                    f"{row['C4']:.4f}",
                    f"{row['R2']:.4f}"
                ])
            
            col_w_p4 = [3.5, 3.1, 2.6, 3.1, 2.6, 2.1]
            fig2.text(2.0 / PAGE_W_CM, 7.6 / PAGE_H_CM,
                      r'4-Parametric Fatigue Model ($FP = C_1 N_f^{C_2} + C_3 N_f^{C_4}$):',
                      fontsize=10.5, weight='bold')
            draw_table(fig2, p4_table, x_cm=2.0, y_top_cm=7.2,
                       col_widths_cm=col_w_p4, row_h_cm=0.50, font_size=9.0)

        # 5. Page 2 Footer (Citation & Page Number Only, NO Acknowledgments)
        if bib_citation:
            fig2.text(0.5 / PAGE_W_CM, 1.10 / PAGE_H_CM, f"Ref: {bib_citation['text']}",
                      fontsize=7.5, ha='left', va='center', fontstyle='italic', color='#222222')
            if bib_citation.get('doi_url'):
                fig2.text(0.5 / PAGE_W_CM, 0.65 / PAGE_H_CM, "DOI: ",
                          fontsize=7.5, ha='left', va='center', color='#222222')
                fig2.text((0.5 + 0.68) / PAGE_W_CM, 0.65 / PAGE_H_CM, bib_citation['doi_url'],
                          fontsize=7.5, ha='left', va='center', color='#0044cc',
                          url=bib_citation['doi_url'])
                          
        fig2.text(20.5 / PAGE_W_CM, 0.65 / PAGE_H_CM, "Page 2/2", fontsize=9.5, ha='right', va='center')

        pdf.savefig(fig2)
        plt.close(fig2)
        
    if sys.platform.startswith('win'):
        try: 
            os.startfile(File_Path)
        except Exception: 
            pass

if __name__ == "__main__":
    create_Preview_Protocol("Protocol.pdf")