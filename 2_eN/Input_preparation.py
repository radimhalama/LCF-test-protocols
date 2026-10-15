# -*- coding: utf-8 -*-
"""
Tool:           LCF & Ratcheting Batch Input File Generator
Description:    Selects output_half.csv files (multi-selection or one-by-one),
                extracts fatigue parameters with console confirmation,
                and generates the 9-row 'Inputs.txt' file for lcf_3d_eval.py.
"""

import os
import math
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd


def get_column_value(df, column_candidates, default=np.nan):
    """Safely extracts a scalar value from the first matching column name."""
    for col in column_candidates:
        if col in df.columns and pd.notna(df[col].iloc[0]):
            return df[col].iloc[0]
    return default


def load_single_output_half(file_path, nf_type="Nf_E10", wp_type="DeltaWp_VarE"):
    """Reads a single output_half.csv and extracts standardized parameters."""
    df = pd.read_csv(file_path)

    sig_a = float(get_column_value(df, ['Sig_A', 'Sig_A (MPa)']))
    eps_a = float(get_column_value(df, ['Eps_A', 'Eps_A (-)']))
    
    # Extract Nf based on selected failure criterion
    if nf_type == "Nf_E10":
        nf_raw = get_column_value(df, ['Nf_E10%', 'Nf_E10'])
    else:
        nf_raw = get_column_value(df, ['Nf', 'Nf_25', 'Nf_25%'])
    
    if pd.isna(nf_raw):
        raise ValueError(f"Life criterion column for '{nf_type}' not found or empty.")
    nf = int(round(float(nf_raw)))

    # Extract plastic work per cycle
    wp_raw = get_column_value(df, [wp_type, f"{wp_type} (MJ/m^3)", f"{wp_type} (MPa)", f"{wp_type} (J/m^3)"])
    if pd.isna(wp_raw):
        raise ValueError(f"Plastic work column for '{wp_type}' not found or empty.")
    wp = float(wp_raw)

    sig_m = float(get_column_value(df, ['Sig_M', 'Sig_M (MPa)'], default=0.0))

    e_ten = float(get_column_value(df, ['E_dyn_tens', 'E_dyn_tens (MPa)']))
    e_com = float(get_column_value(df, ['E_dyn_comp', 'E_dyn_comp (MPa)'], default=e_ten))

    # Averaged dynamic modulus
    e_avg = get_column_value(df, ['E_dyn_avg', 'E_dyn_avg (MPa)'])
    if pd.isna(e_avg) or float(e_avg) <= 0:
        e_avg = 0.5 * (e_ten + e_com)
    else:
        e_avg = float(e_avg)

    folder_name = os.path.basename(os.path.dirname(os.path.abspath(file_path)))
    return {
        'file_path': file_path,
        'folder': folder_name,
        'sig_a': sig_a,
        'eps_a': eps_a,
        'nf': nf,
        'sig_m': sig_m,
        'e_ten': e_ten,
        'e_com': e_com,
        'e_avg': e_avg,
        'wp': wp
    }


def prepare_inputs_txt():
    # 1. Initialize hidden Tkinter window for file picker dialogs only
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("=" * 75)
    print("     BATCH INPUT FILE GENERATOR FOR FATIGUE MODELS (Inputs.txt)")
    print("=" * 75)

    # Criterion selection
    print("Select failure criterion for fatigue life (Nf):")
    print("  1 = Nf_E10 (Cycles to failure - 10% decrease in E_compression)")
    print("  2 = Nf_25  (Number of cycles to failure - 25% drop in Sig_A)")
    crit_choice = input("Enter choice [1]: ").strip() or "1"
    if crit_choice in ["2", "Nf_25", "nf_25"]:
        nf_type = "Nf_25"
    else:
        nf_type = "Nf_E10"
    print(f"  -> Selected failure criterion: {nf_type}\n")

    # Plastic work parameter selection
    print("Select plastic work (per cycle) parameter to import:")
    print("  1 = DeltaWp_VarE (default)")
    print("  2 = DeltaWp")
    print("  3 = Wp_mean")
    wp_choice = input("Enter choice [1]: ").strip() or "1"
    if wp_choice in ["2", "DeltaWp", "deltawp"]:
        wp_type = "DeltaWp"
    elif wp_choice in ["3", "Wp_mean", "wp_mean"]:
        wp_type = "Wp_mean"
    else:
        wp_type = "DeltaWp_VarE"
    print(f"  -> Selected plastic work parameter: {wp_type}\n")

    print("Select file acquisition method:")
    print("  1 = Pick files via dialog (supports Ctrl+Click multi-select or one-by-one)")
    print("  2 = Scan a folder and all its subfolders automatically")

    mode_choice = input("Enter choice [1]: ").strip() or "1"
    selected_files = []

    # 2. File Selection
    if mode_choice == "1":
        while True:
            prompt_title = "Select output_half.csv file(s) [Ctrl+Click to select multiple]" if not selected_files else "Select next output_half.csv file"
            new_files = filedialog.askopenfilenames(
                title=prompt_title,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if new_files:
                for f in new_files:
                    f_abs = os.path.abspath(f)
                    if f_abs not in selected_files:
                        selected_files.append(f_abs)
                        folder_label = os.path.basename(os.path.dirname(f_abs))
                        print(f"  + Added [{len(selected_files)}]: {folder_label} ({os.path.basename(f_abs)})")
            else:
                print("  No file selected in dialog.")

            if not selected_files:
                retry = input("\nNo files added yet. Try again? (Y/N) [Y]: ").strip().upper() or "Y"
                if retry != "Y":
                    print("Aborted.")
                    root.destroy()
                    return
                continue

            # Option to continue adding one-by-one
            add_more = input(f"\nCurrent total files: {len(selected_files)}. Add another file/batch? (y/N) [N]: ").strip().lower() or "n"
            if add_more != "y":
                break

    else:
        root_dir = filedialog.askdirectory(title="Select root directory containing test subfolders")
        if not root_dir:
            print("No folder selected. Aborting.")
            root.destroy()
            return

        print(f"\nScanning '{root_dir}' for output_half.csv files...")
        for subdir, dirs, files in os.walk(root_dir):
            if "output_half.csv" in files:
                f_path = os.path.abspath(os.path.join(subdir, "output_half.csv"))
                selected_files.append(f_path)
                print(f"  + Found: {os.path.basename(subdir)}/output_half.csv")

    if not selected_files:
        print("No files to process. Aborting.")
        root.destroy()
        return

    # 3. Process & Parse Files
    records = []
    print(f"\nParsing {len(selected_files)} test file(s)...")
    for f in selected_files:
        try:
            rec = load_single_output_half(f, nf_type=nf_type, wp_type=wp_type)
            records.append(rec)
        except Exception as e:
            print(f"  [Warning] Skipping '{f}': {e}")

    if not records:
        print("Error: No valid test records could be parsed.")
        root.destroy()
        return

    # 4. Sort tests by fatigue life Nf ascending (Standard in fatigue model fitting)
    records.sort(key=lambda r: r['nf'])

    # 5. Display Review Table in Command Window
    print("\n" + "=" * 85)
    print("LOADED TESTS SUMMARY TABLE (Sorted by life Nf ascending):")
    print("-" * 85)
    print(f"{'#':<3} | {'Folder / Specimen':<20} | {nf_type:<8} | {'Sig_a (MPa)':<11} | {'Eps_a (-)':<10} | {'E_avg (MPa)':<11} | {wp_type:<12}")
    print("-" * 85)
    for idx, r in enumerate(records, start=1):
        print(f"{idx:<3} | {r['folder'][:20]:<20} | {r['nf']:<8} | {r['sig_a']:<11.2f} | {r['eps_a']:<10.5f} | {r['e_avg']:<11.1f} | {r['wp']:<12.4f}")
    print("=" * 85)

    confirm_list = input("\nProceed with these tests? (Y/N) [Y]: ").strip().upper() or "Y"
    if confirm_list != "Y":
        print("Aborted by user.")
        root.destroy()
        return

    # 6. Command Window Inputs for E and Output Directory
    max_e = max([r['e_ten'] for r in records])
    def_E = int(math.ceil(max_e / 1000.0)) * 1000

    print("\n--- PARAMETERS CONFIRMATION (COMMAND WINDOW) ---")
    e_input = input(f"Enter nominal Young's modulus E (MPa) [{def_E}]: ").strip()
    E_modulus = int(e_input) if e_input else def_E

    # Target directory defaults to parent directory of first loaded test
    default_out_dir = os.path.dirname(os.path.dirname(records[0]['file_path']))
    out_dir_input = input(f"Enter save directory [{default_out_dir}]: ").strip()
    save_dir = os.path.abspath(out_dir_input) if out_dir_input else default_out_dir
    os.makedirs(save_dir, exist_ok=True)

    out_name_input = input("Enter output filename [Inputs.txt]: ").strip()
    output_filename = out_name_input if out_name_input else "Inputs.txt"
    if not output_filename.lower().endswith(".txt"):
        output_filename += ".txt"

    # 7. Format Rows
    sig_a_list = [r['sig_a'] for r in records]
    eps_a_list = [r['eps_a'] for r in records]
    nf_list = [r['nf'] for r in records]
    sig_m_list = [r['sig_m'] for r in records]
    e_avg_list = [r['e_avg'] for r in records]
    Wp_list = [r['wp'] for r in records]
    weights = [1.0] * len(records)

    def format_row(values):
        formatted = []
        for v in values:
            if isinstance(v, (int, np.integer)):
                formatted.append(str(v))
            elif isinstance(v, float):
                if abs(v) < 0.1:
                    formatted.append(f"{v:.5f}")
                else:
                    formatted.append(f"{v:.2f}")
            else:
                formatted.append(str(v))
        return ", ".join(formatted)

    # 8. Write the 9-Row Inputs.txt
    output_path = os.path.join(save_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(format_row(sig_a_list) + "\n")        # Row 1: Stress amplitudes
        f.write(format_row(eps_a_list) + "\n")        # Row 2: Strain amplitudes
        f.write(format_row(nf_list) + "\n")           # Row 3: Cycles to failure Nf
        f.write(format_row(weights) + "\n")           # Row 4: Weights (1.0)
        f.write(f"{int(E_modulus)}\n")                # Row 5: Single dynamic modulus E
        f.write(format_row(e_avg_list) + "\n")        # Row 6: Averaged dynamic modulus (E_avg)        
        f.write(", ".join([f"{v:.5f}" for v in Wp_list]) + "\n")   # Row 7: Plastic work (5 decimal places)
        f.write(format_row(sig_m_list) + "\n")        # Row 8: Mean stresses        
        f.write(f"{nf_type}\n")                       # Row 9: Selected criterion string (Nf_E10 or Nf_25)

    print("\n" + "=" * 75)
    print(f"SUCCESS: File successfully created:")
    print(f"  --> {output_path}")
    print(f"Total processed tests: {len(records)} (sorted by life {nf_type} ascending)")
    print("=" * 75)

    root.destroy()


if __name__ == "__main__":
    prepare_inputs_txt()