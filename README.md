# Energy Profiling & Repair Toolkit

This repository provides an **end-to-end solution** for analyzing Python code performance and energy usage, then automatically suggesting and testing optimizations. It includes:

- **`energy_profile.py`** – Collects energy usage, CPU/memory usage, and code quality metrics from datasets like HumanEval or EvalPlus.  
- **`energy_repair.py`** – Reads profiling data, generates “energy repairs” (via an AI endpoint), and re-tests/benchmarks to evaluate improvements.  
- **`dashboard.py`** – Provides a Dash web interface to visualize results (plots, correlation analyses, code comparisons, etc.).  
- **Test suites** – **`test_energy_profile.py`** and **`test_energy_repair.py`** contain comprehensive unit tests.

---

## Table of Contents

1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Setup & Installation](#setup--installation)  
4. [Usage](#usage)  
   - [1. Profiling with `energy_profile.py`](#1-profiling-with-energy_profilepy)  
   - [2. Generating Repairs with `energy_repair.py`](#2-generating-repairs-with-energy_repairpy)  
   - [3. Visualizing with `dashboard.py`](#3-visualizing-with-dashboardpy)  
5. [Further Explanation](#further-explanation)  
   - [A. How `energy_profile.py` Works](#a-how-energy_profilepy-works)  
   - [B. How `energy_repair.py` Works](#b-how-energy_repairpy-works)  
   - [C. Dashboard Overview](#c-dashboard-overview)  
6. [Testing](#testing)  
7. [Troubleshooting](#troubleshooting)  
8. [Contributing](#contributing)  
9. [License](#license)

---

## Overview

**Primary Goal:**  
Measure and optimize the energy efficiency of Python functions. This includes:

- Profiling CPU & memory usage along with energy consumption (via RAPL on Linux or CPU-based estimates otherwise).  
- Evaluating code quality (e.g., cyclomatic complexity, maintainability, pylint score).  
- Storing results in CSV files and visualizing them through both static plots and an interactive Dash-based dashboard.  
- Attempting automated “repairs” (optimizations) with an AI-driven approach, then re-checking performance and correctness.

---

## Key Features

1. **Energy Profiling**  
   - **Intel RAPL** support for direct CPU power measurement on Linux.  
   - Fallback to approximate CPU usage measurements on other systems.

2. **Code Quality Analytics**  
   - Computes cyclomatic complexity with [Radon](https://github.com/rubik/radon).  
   - Runs **pylint** checks with a custom config to derive a code rating.  
   - Calculates a maintainability score based on radon’s maintainability index or custom heuristics.

3. **Visualization**  
   - Matplotlib/Seaborn charts for quick data overviews (bar plots, scatter plots, heatmaps).  
   - **`dashboard.py`** for interactive exploration of improvements, correlations, code diffs, etc.

4. **Automated “Repairs”**  
   - AI-based function rewriting via an HTTP endpoint (`http://localhost:11434/api/generate` by default).  
   - Validate each repair by re-running the test suite and measuring new energy usage.  
   - Outputs best solution to a new CSV, detailing improvement in energy, CPU, memory, and time.

5. **Testing Suite**  
   - **`test_energy_profile.py`** and **`test_energy_repair.py`** provide a broad set of unit tests for reliability.

---

## Setup & Installation

1. **Clone This Repository**

   ```bash
   git clone https://github.com/your-username/energy-profiling-repair.git
   cd energy-profiling-repair
   ```

2. **Create a Virtual Environment (Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *Note:*  
   - If RAPL is unsupported by your machine, the scripts automatically switch to CPU-based measurements.  
   - [Perun](https://pypi.org/project/perun/) sometimes requires manual installation from GitHub.  
   - Additional libraries might be needed for advanced usage of `datasets`.

4. **Optional**: Ensure you have a local AI model endpoint running on port 11434 if you plan to use code auto-repair.

---

## Usage

### 1. Profiling with `energy_profile.py`

```bash
python energy_profile.py --dataset humaneval
```
OR
```bash
python energy_profile.py --dataset evalplus
```

- **Loads** the selected dataset from Hugging Face.  
- **Runs** each snippet through an energy measurement routine (including CPU/memory usage).  
- **Analyzes** cyclomatic complexity, code maintainability, and pylint rating.  
- **Outputs** a CSV file (e.g., `human_eval_profile.csv`) and a folder of Seaborn/Matplotlib charts.

### 2. Generating Repairs with `energy_repair.py`

```bash
python energy_repair.py
```

- **Reads** the CSV created by `energy_profile.py` (e.g. `evalplus_profile.csv`).  
- **Calls** your AI model endpoint with code, metrics, and best practices.  
- **Re-measures** energy usage and test correctness for each proposed fix.  
- **Saves** final suggestions in a new CSV (like `repair_suggestions_YYYYMMDD_HHMMSS.csv`).

### 3. Visualizing with `dashboard.py`

```bash
python dashboard.py
```

- **Launches** a Dash web server on `http://127.0.0.1:8050` by default.  
- **Interactive** dashboards to filter data, see code diffs, and analyze correlations or statistical tests.  
- **Supports** combining multiple CSV files in the `data/` folder (e.g., RAPL vs. powermetrics comparisons).

---

## Further Explanation

### A. How `energy_profile.py` Works

- **Module Imports & Logging**: Sets up python logging, imports `psutil`, `radon`, `pylint`, etc.  
- **Key Functions**:  
  - `profile_code`: Executes code in a restricted namespace, records CPU/memory usage, attempts RAPL or fallback measurements, and returns structured metrics.  
  - `analyze_code_quality`: Creates a temp file, runs pylint, calculates maintainability via radon.  
  - `run_human_eval_function` & `run_mbpp_function`: Specialized test harnesses for each dataset.  
- **Main Routine**: Uses `argparse` to handle `--dataset humaneval/evalplus`, then loops over each function/test pair in the dataset.

### B. How `energy_repair.py` Works

- **OptimizationHistoryBuffer**: Tracks past improvements, best practices, and code transformations.  
- **`generate_repair(code_snippet, energy_profile_data, ...)`**:  
  1. Profiles the snippet again for a baseline.  
  2. Builds an AI prompt with code and metrics.  
  3. Submits the prompt to the local AI endpoint.  
  4. Parses any returned “repairs,” re-tests them, re-measures energy usage.  
  5. Chooses the best improvement, logs it.  
- **Test Integration**:  
  - `run_test_cases`: Writes code + test code to temp files, imports and runs them.  
- **Final CSV**: Summarizes each successful fix with improvements in energy/time/CPU usage.

### C. Dashboard Overview

- **Dash + Plotly**: Provides advanced, interactive visualizations.  
- **Data Loading**: Gathers multiple CSVs from a `data/` folder or any path.  
- **Code Comparison**: Highlights function differences (original vs optimized) using AST comparisons.  
- **Statistical Tests**: Includes Kruskal-Wallis, bootstrap confidence intervals, correlation heatmaps, and more to examine performance/energy differences.

---

## Testing

Both `energy_profile.py` and `energy_repair.py` have their own test suites:

- **`test_energy_profile.py`** – Mocks RAPL calls, verifies CPU/memory usage logic, tests code extraction and complexity.  
- **`test_energy_repair.py`** – Checks repair generation logic, partial mocking of AI calls, test code correctness, etc.

Run all tests via:

```bash
python -m unittest discover .
```

---

## Troubleshooting

1. **RAPL Unavailable**  
   - If `/sys/class/powercap/intel-rapl` isn’t found (e.g. on Mac or older hardware), the scripts default to CPU-based estimations.  

2. **AI Endpoint Issues**  
   - Make sure your local AI service is live if you want to generate code fixes.  
   - Adjust the URL in `energy_repair.py` if needed.

3. **Visualization Problems**  
   - Missing Plotly or Dash packages? Re-install from `requirements.txt`.  
   - Seaborn/Matplotlib warnings occur if you’re missing optional dependencies.

4. **Permission Errors**  
   - On Linux, reading Intel RAPL data typically requires root/sudo privileges.

---

## Contributing

1. **Fork** this repo and create a feature branch (`git checkout -b feature/XYZ`).  
2. Commit changes and **push** to your branch.  
3. **Open** a pull request and describe your improvements in detail.

All contributions that enhance energy measurement accuracy, code analysis, or the AI-based approach are welcome!