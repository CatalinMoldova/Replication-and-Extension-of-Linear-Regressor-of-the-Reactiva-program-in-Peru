

# Replication and Extension of Linear Regressor of the Reactiva Program in Peru

This repository replicates and extends a linear regression model used to study credit risk and firm outcomes under the Reactiva Peru program, a large-scale government-guaranteed loan initiative launched during the COVID-19 pandemic. The project focuses on understanding how firm characteristics relate to credit risk and loan performance, and compares the baseline linear model with alternative specifications and regularized methods.[1][2][3]

## Project Overview

- **Goal**: Reproduce a baseline linear regression model for Reactiva Peru credit risk or firm performance and extend it with additional features, regularization, and robustness checks.[2][4][1]
- **Context**: Reactiva Peru provided guaranteed loans through the banking system to protect firms’ liquidity and employment during the COVID-19 shock.[5][4][3]
- **Methods**: Ordinary Least Squares (OLS) as the core model, with extensions using ridge and lasso regression and alternative specifications/matching strategies inspired by related literature.[4][1][2]

The code is structured to allow transparent replication of the original specification and modular experimentation with additional variables and models.

## Repository Structure

Adapt the paths and names if they differ in your repo:

- `data/`  
  - Raw and/or cleaned firm-level and loan-level data used in the regressions (not committed if sensitive).  
- `notebooks/`  
  - Exploratory data analysis and step‑by‑step replication of the original regression specification.  
- `src/`  
  - Reusable scripts for data cleaning, feature engineering, model estimation, and plotting.  
- `results/`  
  - Saved regression outputs (tables), diagnostics, and figures.  
- `docs/`  
  - Notes on the original paper, modeling choices, and interpretation of results.  

If your local layout differs, update this section to match your actual folders and filenames.

## Installation and Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/CatalinMoldova/Replication-and-Extension-of-Linear-Regressor-of-the-Reactiva-program-in-Peru.git
   cd Replication-and-Extension-of-Linear-Regressor-of-the-Reactiva-program-in-Peru
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   If the project uses a `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

   If the project uses `environment.yml` (conda):
   ```bash
   conda env create -f environment.yml
   conda activate reactiva-peru
   ```

4. **Place data**

   - Add the necessary input datasets into the `data/` folder (or the path expected by the notebooks and scripts).  
   - Due to confidentiality, the original administrative datasets used in Reactiva Peru studies are not included; users must obtain or simulate compatible data structures.[6][5][4]

## How to Run the Analysis

### 1. Replication of the Baseline Linear Model

1. Open the main replication notebook (e.g. `notebooks/01_replication_linear_model.ipynb`).  
2. Configure any data paths at the top of the notebook.  
3. Run all cells to:
   - Load and clean the data.  
   - Reconstruct the variables used in the baseline specification (e.g., credit risk indicators, firm size, leverage, sector dummies).  
   - Estimate the linear regression model and reproduce core tables/figures from the reference study.[1][2][4]

### 2. Extensions and Robustness

Use the extension notebooks (e.g. `notebooks/02_extension_ridge_lasso.ipynb`, `notebooks/03_robustness_checks.ipynb`) to:

- Add new features (e.g., additional balance sheet ratios, interaction terms, or temporal dummies).  
- Fit ridge and lasso models and compare performance with the baseline OLS in terms of predictive accuracy and stability of coefficients.[2][1]
- Run alternative specifications (e.g., different samples, matching strategies, fixed effects) inspired by the Reactiva Peru and credit‑guarantee literature.[5][4][6][1]

### 3. Exporting Results

Scripts or notebook cells are typically provided to:

- Export regression tables (e.g., to LaTeX or CSV) into `results/`.  
- Save plots summarizing coefficient estimates, prediction performance, or distributional outcomes.

Adjust filenames and output paths as needed for your workflow.

## Methods and References

The project is informed by empirical work on Reactiva Peru and related credit‑guarantee programs, as well as literature comparing penalized and classical regression for predictive modeling:

- Studies on Reactiva Peru’s impact on credit, employment, bank risk, and firm outcomes.[7][3][4][6][1][5]
- Machine learning for credit risk in the Reactiva Peru program, with emphasis on lasso and ridge.[1][2]
- Broader discussions of government credit guarantees and post‑COVID firm financing.[8][2]

Please acknowledge these sources and this repository if you build upon the code or results in academic or professional work.

## License

State the license under which this code is released (e.g. MIT, Apache 2.0, or “All rights reserved”). Make sure the chosen license is compatible with any data access and usage restrictions.

## Disclaimer

This repository is for research and educational purposes. It does not provide financial advice or official evaluation of the Reactiva Peru program, and it respects the intellectual property and confidentiality constraints of the underlying data and original studies.[4][6][5][1]
