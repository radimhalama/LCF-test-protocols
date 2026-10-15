# LCF Evaluation Toolchain

[![FABER COST Action CA23109](https://faber-cost.eu/wp-content/uploads/2025/01/logo.png)](https://faber-cost.eu/)

Open-source Python framework developed within the **[FABER network (COST Action CA23109 - WG 4.7 LCF)](https://faber-cost.eu/)** for advanced fatigue data processing, constitutive parameter identification, and standardized test protocol export in low-cycle fatigue domain.

## Repository Architecture

### 1. Single-Specimen Evaluation (`1_LCF_UNI`)
Post-processes continuous raw time-series data exported from universal testing machines (LabControl, Instron, MTS):
- Automated cycle segmentation and hysteresis loop extraction.
- Feature extraction at stabilized half-life ($N_f / 2$): $\sigma_a$, $\varepsilon_{ap}$, directional dynamic moduli ($E_{\text{upp}}, E_{\text{btm}}$), and plastic work ($W_p$).
- Failure identification using physical criteria ($N_f^{25}$ load drop and $N_f^{E10}$ stiffness drop).
- Automated generation of 2-page single-specimen PDF protocols (`Protocol_LCF.py` and `Protocol_Ratcheting.py`).

### 2. Multi-Specimen $\varepsilon$ - $N$ Modeling (`2_eN_UNI/`)
Ingests the consolidated 9-row `Inputs.txt` record to perform constitutive parameter identification:
- **Compatible 3D Method:** Singular Value Decomposition (SVD) in $\log_{10}(\varepsilon_{ap}) - \log_{10}(\sigma_a) - \log_{10}(2N_f)$ space enforcing $n' = b/c$ and $K' = \sigma_f' / (\varepsilon_f')^{n'}$.
- **Conventional Method:** Independent Basquin, Manson–Coffin, and Ramberg–Osgood regressions.
- Evaluates properties across both constant ($E_{\text{const}}$) and dynamic ($E_{\text{var}}$) elastic moduli.
- Non-linear 4-parametric energy modeling for $\text{SWT}$ and $W_p$ ($FP = C_1 N_f^{C_2} + C_3 N_f^{C_4}$).
- Generates a publication-ready, 2-page PDF test protocol (`Protocol.py`) with clickable DOI citations.

## Documentation & Examples

- **User Manual:** [Read the User Manual (PDF)](docs/FABER_LCF_Toolchain_User_Manual.pdf) 
- **Sample Output 1 - LCF:** [View a Pre-Generated LCF Test Protocol (PDF)](docs/Protocol_LCF.pdf) 
- **Sample Output 2 - Ratcheting:** [View a Pre-Generated Ratcheting Test Protocol (PDF)](docs/Protocol_Ratcheting.pdf) 
- **Sample Output 3 - 3D method:** [View a Pre-Generated 3D method Protocol (PDF)](docs/Protocol_eN.pdf) 

## Installation

```bash
git clone https://github.com/radimhalama/LCF-test-protocols.git
cd LCF-test-protocols
pip install -r requirements.txt
