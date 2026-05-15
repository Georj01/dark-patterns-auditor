# Dark Patterns Auditor

> A quantitative analysis tool to identify and evaluate dark patterns and psychological biases in digital interfaces, with a case study focus on video games.

## 📌 Overview
This repository contains a Python-based analytical tool developed as part of my final degree project (TFG). It audits user interfaces to detect Dark Patterns and maps them directly to the cognitive biases they exploit. 

While the provided dataset focuses on *Fortnite*, the script is entirely platform-agnostic. It processes observational data to generate a comprehensive suite of statistical reports, cross-tabulations, and visualizations, turning subjective UX audits into quantifiable, data-driven insights.

## 🚀 Features
- **Data Validation & Cleaning**: Automatically handles missing values and validates mandatory fields (`categoria_patron`, `sesgo_psicologico`).
- **Descriptive Statistics**: Generates frequency counts and combination matrices of patterns and biases.
- **Association Rules**: Calculates Support, Confidence, and Lift to uncover the strongest correlations between specific UI patterns and psychological triggers.
- **Hypothesis Testing**: Performs Chi-Square tests of independence, outputting expected matrices, adjusted residuals, p-values, and Cramer's V for effect size.
- **Automated Data Visualization**: Exports horizontal bar charts and correlation heatmaps using Seaborn and Matplotlib. Includes automated segmentation for visual rhetoric (UI colors and element sizes) and sound alerts.
- **Business-Ready Reporting**: Outputs all cross-tabulations and statistical matrices to Excel (`.xlsx`) and CSV formats for easy integration into academic papers or stakeholder presentations.

## 🛠️ Tech Stack
- **Python 3.8+**
- **Pandas** (Data manipulation & Cross-tabulation)
- **NumPy** (Numerical operations)
- **SciPy** (Statistical hypothesis testing)
- **Matplotlib & Seaborn** (Data visualization)

## 📦 Installation
1. Clone the repository:
   `git clone https://github.com/TU-USUARIO/dark-patterns-auditor.git`
2. Navigate to the folder:
   `cd dark-patterns-auditor`
3. Install the required dependencies:
   `pip install pandas numpy scipy matplotlib seaborn xlsxwriter`

## ⚙️ Usage
Run the analysis script via the command line. By default, it reads `Datos.csv` in the root directory and outputs all reports to an `output/` folder.

`python Analisis.py --input Datos.csv --output output`

### Arguments:
- `--input`: Path to your input CSV file (must use `;` as a delimiter). Default is `Datos.csv`.
- `--output`: Directory where all generated reports and graphs will be saved. Default is `output/`.

## 🗄️ Expected Data Format
The input CSV should be separated by semicolons (`;`) and must include at least the following columns to run successfully:
- `categoria_patron`: The identified dark pattern (e.g., *False_Urgency*, *Sunk_Cost*).
- `sesgo_psicologico`: The cognitive bias being exploited (e.g., *FOMO*, *Falacia_Coste_Hundido*).

Optional columns that unlock specific visual charts and segmentations:
- `retorica_color`: Colors used in the UI.
- `retorica_tamano`: Sizes of the UI elements.
- `alerta_sonora`: Presence of sound alerts.
- `videojuego`: The specific game analyzed (generates per-game segmented reports).
- `modelo`: Monetization model (generates per-model segmented reports).

*Note: Any additional categorical columns (e.g., `ubicacion_ui`) are automatically detected by the script and normalized cross-tabulations will be generated for them.*

## 📊 Outputs Generated
Upon execution, the script generates an `output/` directory containing:
- `chi2_test.txt`: Detailed breakdown of the Chi-Square test.
- `analisis_estadistico.xlsx`: Multi-sheet Excel file containing observed frequencies, expected frequencies, residuals, and adjusted residuals.
- `asociacion_patron_sesgo.csv`: Association rules sorted by Lift.
- `*.png`: High-resolution bar charts and heatmaps.
- Multiple `crosstab_*.csv`: Normalized cross-tabulations for every categorical variable against the detected dark patterns.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
