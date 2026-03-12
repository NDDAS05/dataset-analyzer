<div align="center">

# 🔬 Automated Dataset Analyzer

### *A Mini Automated Data Scientist — built entirely in Python*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![pandas](https://img.shields.io/badge/pandas-Data%20Handling-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Engine-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![matplotlib](https://img.shields.io/badge/matplotlib-Visualization-11557C?style=for-the-badge)](https://matplotlib.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**Drop in any CSV. Get a full interactive EDA report in seconds — auto-launched in your browser.**

[Features](#-features) · [Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Report Walkthrough](#-report-walkthrough) · [Project Structure](#-project-structure) · [Tech Stack](#-tech-stack)

---

![Report Preview](https://img.shields.io/badge/Report-Auto%20Generated%20HTML-6C8EF5?style=flat-square)
![Charts](https://img.shields.io/badge/Charts-Per--Feature%20Individual-52C17E?style=flat-square)
![ML](https://img.shields.io/badge/ML-Auto%20Task%20Detection-F4845F?style=flat-square)

</div>

---

## What Is This?

This project is a **fully automated exploratory data analysis (EDA) pipeline** that works on any CSV dataset. You run one command, and it handles everything:

- Loads and validates your data
- Runs comprehensive quality checks
- Generates full-size, per-feature visualizations
- Auto-detects and trains a machine learning model
- Packages everything into a **beautiful, self-contained HTML report**
- Opens that report **automatically in your browser**

No Jupyter notebooks. No manual configuration. No boilerplate. Just results.

---

## Features

### Automated EDA
- **Dataset overview** — shape, dtypes, memory usage, column type breakdown
- **Descriptive statistics** — count, mean, std, min/max, quartiles for every numeric column
- **Smart column detection** — automatically identifies numeric, categorical, and datetime columns

### Data Quality Diagnostics
- **Missing value analysis** — per-column count, percentage, and severity rating (Low / Medium / High)
- **Duplicate detection** — flags exact duplicate rows with count and percentage
- **Outlier detection** — IQR method applied to every numeric column, with computed bounds

### Visualizations
- **Per-feature distribution histograms** — one full-size chart per numeric column with mean & median reference lines
- **Interactive tabbed viewer** — switch between features with a single click in the report
- **Correlation heatmap** — lower-triangle seaborn heatmap with diverging palette
- **Categorical bar charts** — per-feature, showing top 10 categories with count labels
- **Missing values chart** — horizontal bar chart sorted by missingness
- **Outlier chart** — per-column outlier percentage visualization

### Machine Learning Insights
- **Auto task detection** — classifies the problem as classification or regression based on target cardinality
- **Auto target detection** — detects target column by name (`target`, `label`, `class`, `y`) or falls back to last column
- **Random Forest model** — trains `RandomForestClassifier` or `RandomForestRegressor` automatically
- **Metrics reported:**
  - Classification → Accuracy, Confusion Matrix
  - Regression → R², MAE, RMSE
- **Feature importance chart** — ranked bar chart of top 15 features
- **Feature importance table** — with visual bar indicators per feature

### Auto-Generated Insights
- Natural-language bullet points summarizing key findings
- Covers missing data, duplicates, outliers, correlations, and ML results

### Report & UX
- **Single self-contained HTML file** — all charts embedded as base64, no external dependencies
- **Auto browser launch** — report opens immediately after generation
- **Per-dataset output folders** — `reports/<dataset_name>/` so multiple runs never overwrite each other
- **Light, readable design** — Nunito font, soft color palette, rounded cards, smooth sidebar navigation
- **Responsive layout** — works on any screen size

---

## Quick Start

There are **two ways** to use this tool — a hosted web app and a local CLI.

---

### Option A — Web App (Recommended for sharing)

**[ Try it live on Streamlit Cloud](https://your-app.streamlit.app)**  
No installation. Upload your CSV, get your report instantly.

Or run locally:
```bash
git clone https://github.com/nirupam-das/dataset-analyzer.git
cd dataset-analyzer
pip install -r requirements.txt
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

---

### Option B — CLI Tool (Install once, run anywhere)

```bash
git clone https://github.com/nirupam-das/dataset-analyzer.git
cd dataset-analyzer
pip install -e .
```

This registers the `analyze` command globally. Then from anywhere:

```bash
analyze path/to/your/dataset.csv
```

Your browser opens with the full report automatically.

---

## ⚙️ CLI Usage & Options

```bash
analyze <csv_file> [options]
```

| Argument | Description | Example |
|---|---|---|
| `csv_file` | Path to the input CSV *(required)* | `data.csv` |
| `--target COL` | Specify the ML target column | `--target price` |
| `--no-ml` | Skip the machine learning step | `--no-ml` |
| `--no-launch` | Don't auto-open the browser | `--no-launch` |

### Examples

```bash
# Full analysis — run from anywhere on your system
analyze ~/Downloads/titanic.csv

# Specify your target column explicitly
analyze ~/datasets/house_prices.csv --target SalePrice

# EDA only, no ML (faster)
analyze pokemon.csv --no-ml
```

> **Where does the report go?**
> A `reports/` folder is created in whichever directory you run the command from.

---

## 🌐 Streamlit Web App

The web app (`app.py`) gives the same analysis through a browser interface — perfect for sharing with non-technical users or showcasing on a resume.

```bash
streamlit run app.py
```

### Web App Features
- **Drag & drop CSV upload** — no file paths needed
- **8 interactive tabs** — each section is a dedicated tab
- **Dropdown feature selectors** — pick any column to inspect its chart
- **Live ML training** — toggle on/off from the sidebar, specify target column
- **One-click HTML download** — generates and downloads the full report

### Deploy to Streamlit Cloud (Free)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set **main file** to `app.py`
4. Click Deploy — you get a public URL in ~2 minutes

---

## How It Works

```
CSV File
   │
   ▼
┌─────────────────────────────────────────────────────┐
│  1. LOAD          pandas.read_csv() with encoding   │
│                   fallback (utf-8 → latin-1 → cp1252)│
├─────────────────────────────────────────────────────┤
│  2. ANALYZE       Overview, missing values,         │
│                   duplicates, outliers (IQR),        │
│                   correlations, value counts         │
├─────────────────────────────────────────────────────┤
│  3. VISUALIZE     Per-feature histograms,           │
│                   heatmap, bar charts — saved as PNG │
├─────────────────────────────────────────────────────┤
│  4. ML INSIGHTS   Auto-detect task → preprocess →   │
│                   RandomForest → metrics + importances│
├─────────────────────────────────────────────────────┤
│  5. INSIGHTS      Natural language summary bullets  │
├─────────────────────────────────────────────────────┤
│  6. REPORT        Jinja2 renders HTML, all images   │
│                   embedded as base64 data URIs       │
├─────────────────────────────────────────────────────┤
│  7. LAUNCH        webbrowser.open() fires report    │
└─────────────────────────────────────────────────────┘
   │
   ▼
reports/<dataset>/dataset_report.html  ← opens in browser
```

---

## Report Walkthrough

The generated HTML report has a **fixed sidebar navigation** with 7 sections:

| Section | What's inside |
|---|---|
| 🗂️ **Overview** | Row/column count, KPI cards, column type pills, full descriptive stats table |
| 🔍 **Data Quality** | Missing values table + chart, duplicate count, outlier table + chart |
| 📈 **Distributions** | Tabbed viewer — one full-size histogram per numeric feature |
| 🔗 **Correlations** | Heatmap + list of strong correlations (&#124;r&#124; ≥ 0.7) |
| 🏷️ **Categorical** | Tabbed viewer — one full-size bar chart per categorical feature |
| 🤖 **ML Insights** | Metrics, feature importance chart + ranked table, confusion matrix |
| 💡 **Insights** | Numbered auto-generated findings covering the entire analysis |

> **Tabbed Viewers** — instead of cramped grids, Distributions and Categorical sections each have a tab bar. Click any column name to instantly view its full-size chart.

---

## Project Structure

```
dataset-analyzer/
│
├── main.py                   ← Entry point — run this
├── requirements.txt          ← Python dependencies
├── README.md
│
├── reports/                  ← All generated reports live here
│   └── <dataset_name>/
│       ├── dataset_report.html   ← The final self-contained report
│       └── _assets/              ← Chart PNGs (referenced by report)
│
└── src/                      ← Internal engine (no need to modify)
    ├── analyzer.py           ← Data loading, EDA, quality checks, insights
    ├── visualizer.py         ← All chart generation (matplotlib + seaborn)
    ├── ml_insights.py        ← ML pipeline (preprocessing, training, metrics)
    ├── report_generator.py   ← Jinja2 HTML rendering + base64 image embedding
    └── templates/
        └── report_template.html  ← Full HTML/CSS/JS report template
```

### Module Responsibilities

**`main.py`** — Orchestrator. Parses CLI args, calls each module in sequence, handles errors, auto-launches browser.

**`src/analyzer.py`** — Loads CSV with encoding fallback. Computes dataset overview, descriptive stats, missing values, duplicates, IQR outliers, correlations, and value counts. Generates natural-language insight strings.

**`src/visualizer.py`** — Generates all charts using matplotlib and seaborn with a consistent light design system. Each feature gets its own individual full-size PNG. Supports feature importance and confusion matrix charts for ML.

**`src/ml_insights.py`** — Detects task type (classification vs regression), preprocesses features (encoding, imputation), trains a Random Forest, returns metrics and feature importances.

**`src/report_generator.py`** — Embeds all PNGs as base64 data URIs, builds the Jinja2 template context, renders the final HTML, and writes it to disk.

---

## Tech Stack

| Library | Role |
|---|---|
| **pandas** | Data loading, cleaning, aggregation |
| **numpy** | Numerical operations, outlier bounds |
| **matplotlib** | All chart rendering |
| **seaborn** | Heatmap and color palette utilities |
| **scikit-learn** | Random Forest, preprocessing, metrics |
| **Jinja2** | HTML report templating |
| **webbrowser** | Auto-launching the report in the browser |

**No external web frameworks. No databases. No API keys. Pure Python.**

---

## Supported Dataset Types

The analyzer handles a wide variety of CSV structures automatically:

- ✅ Numeric-heavy datasets (regression targets, sensor data, financial data)
- ✅ Mixed numeric + categorical datasets (surveys, customer data, HR data)
- ✅ Classification datasets (Titanic, Iris, Pokemon types, etc.)
- ✅ Datasets with missing values, duplicates, and outliers
- ✅ Multiple encoding formats (UTF-8, Latin-1, CP1252, ISO-8859-1)
- ✅ Large files (tested up to 100k+ rows)

---

## Sample Output Structure

After running `python main.py titanic.csv`, the project looks like:

```
dataset-analyzer/
├── main.py
├── reports/
│   └── titanic/
│       ├── dataset_report.html    ← opened automatically in browser
│       └── _assets/
│           ├── dist_Age.png
│           ├── dist_Fare.png
│           ├── dist_Pclass.png
│           ├── cat_Sex.png
│           ├── cat_Embarked.png
│           ├── correlation_heatmap.png
│           ├── missing_values.png
│           ├── outliers.png
│           ├── feature_importance.png
│           └── confusion_matrix.png
└── src/
    └── ...
```
---

## 👤 Author

**Nirupam Das**

Built as a demonstration of end-to-end Python engineering — combining data analysis, machine learning, visualization, templating, and CLI design into a single automated pipeline.

---

</div>
