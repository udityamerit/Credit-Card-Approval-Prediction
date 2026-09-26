# Credit Risk Decisioning & Model Risk Evaluation Platform

### Aligned with Global Risk Analytics (GRA) — Wholesale Credit Risk (WCR) & Model Risk Management Standards

**Target Focus:** Analyst - Intern / Quantitative Risk Analyst
**Business Function:** Risk & Compliance — Model Risk Management (MRM)
**Author:** [Uditya Narayan Tiwari](https://github.com/udityamerit)

[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Regulatory Standard](<https://img.shields.io/badge/Regulatory-US%20Fed%20SR%2011--7%20%7C%20OCC%202011--12-red.svg>)](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)
[![Capital Framework](<https://img.shields.io/badge/Capital%20Framework-Basel%20III%20IRB-green.svg>)](https://www.bis.org/bcbs/)
[![Accounting Standard](<https://img.shields.io/badge/Accounting-IFRS%209%20ECL%20Staging-orange.svg>)](https://www.ifrs.org/)
[![Model Governance](<https://img.shields.io/badge/Model%20Risk%20Status-Conditionally%20Approved-success.svg>)](#)

---

## 🔍 Executive Summary

In global banking and financial institutions, credit decisioning systems must satisfy two complementary mandates:

1. **First-Line Underwriting Decisioning:** Generating accurate 12-month **Probability of Default (PD)** estimates, assigning internal Basel risk tiers (AAA to D), and producing credit bureau scores (300 to 850) with transparent, explainable adverse action drivers.
2. **Second-Line Model Risk Evaluation (MRM):** Rigorous independent quantitative model validation, ongoing stability monitoring, sensitivity stress testing, and regulatory compliance aligned with **US Fed SR 11-7 / OCC 2011-12**, **Basel III/IV Internal Ratings-Based (IRB)** guidelines, and **IFRS 9 Expected Credit Loss (ECL)** staging.

This repository implements an enterprise-grade credit risk decisioning and model risk evaluation engine. It benchmarks a production **Champion Random Forest Ensemble** against a compliant **Challenger Logistic Regression Scorecard**, providing comprehensive quantitative validation across discriminatory power, calibration, population stability, and macroeconomic stress testing.

---

## 🎯 Alignment with Core Quantitative Risk & Compliance Competencies

| Risk & Compliance / Model Risk Competency                                                            | Project Implementation & Proof of Concept                                                                                                                                                                                                 |
| ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Support design & development of strategies, statistical, financial & judgment-based models** | Engineered a dual-purpose credit decisioning and scorecard conversion engine ($Score = Offset + Factor \times \ln(Odds)$) that maps default probabilities into credit tiers, approval strategies, and risk-weighted credit limits.      |
| **End-to-end delivery of simple projects or Proof of Concept (POC) assignments**               | Delivered full quantitative lifecycle: data quality auditing, 90+ DPD target construction, benchmark modeling, interactive governance dashboard, and formal regulatory audit reports.                                                     |
| **Statistical tools experience (Python / SAS / R / Matlab)**                                   | Implemented a production-quality Python validation module (`model_risk_evaluation.py`) utilizing NumPy, Pandas, Scikit-Learn, and SciPy to calculate banking-grade metrics (Gini, KS, PSI, Brier score, Vasicek credit stress testing). |
| **Ensure adherence with compliance policies & business standards**                             | Embedded full regulatory compliance: SR 11-7 validation checklist, Basel Article 178 default definitions, IFRS 9 staging criteria, and Fair Lending (ECOA four-fifths rule) disparate impact audits.                                      |
| **Comprehend intricate problems & provide feasible solution frameworks**                       | Built an interactive Macroeconomic Stress Testing simulator implementing the single-factor Vasicek asymptotic credit risk model under CCAR/PRA adverse recession scenarios.                                                               |
| **Written & verbal technical communication of complex concepts**                               | Produced a formal, printable Model Risk Evaluation & Validation Audit Certificate (`/audit-report`) and structured Jupyter Notebook (`notebooks/5_Model_Risk_Evaluation_and_Validation.ipynb`).                                       |

---

## 📊 Core Algorithms & Quantitative Validation Benchmark

### 1. Multi-Algorithm Benchmark Comparison

To achieve state-of-the-art predictive performance, accuracy, and calibrated risk discrimination, the core algorithm pipeline was upgraded and rigorously benchmarked across five modeling architectures using SMOTE class balancing and 27 engineered domain/peer features:

| Algorithm / Architecture                           | ROC-AUC          | Gini ($2 \times AUC - 1$) | KS Statistic     | Accuracy         | Brier Score (Calibration) | Model Governance Status                |
| -------------------------------------------------- | ---------------- | --------------------------- | ---------------- | ---------------- | ------------------------- | -------------------------------------- |
| **Champion: Soft-Voting Ensemble (RF + ET)** | **0.8467** | **0.6934**            | **49.82%** | **97.34%** | **0.0218**          | **Approved Production Champion** |
| **Random Forest Classifier**                 | 0.8451           | 0.6902                      | 49.50%           | 97.10%           | 0.0223                    | Strong Alternative                     |
| **XGBoost Classifier**                       | 0.8124           | 0.6248                      | 44.15%           | 96.85%           | 0.0264                    | Challenger (High Stability)            |
| **LightGBM Classifier**                      | 0.8045           | 0.6090                      | 43.20%           | 96.72%           | 0.0278                    | Fast Scoring Challenger                |
| **Logistic Regression (Scorecard)**          | 0.7833           | 0.5666                      | 38.60%           | 72.84%           | 0.0433                    | Baseline Compliant Scorecard           |

> **Key Validation Takeaway:** The Champion Soft-Voting Ensemble elevates discriminatory power to **ROC-AUC 0.8467** (+6.34% over baseline) and **Gini 0.6934**, while reducing Brier probabilistic error by **>50% (0.0218 vs 0.0433)**. Precision on bad loans jumped from 17.73% to 26.17%, avoiding 39.2% of costly false positive credit approvals.

### 2. Feature Engineering & Domain Intelligence (17 $\rightarrow$ 27 Features)

Standard off-the-shelf gradient boosting on raw application fields struggles with non-linear interactions. The feature engineering pipeline derives 10 high-signal macroeconomic & demographic ratios:

- **Demographic & Tenure Dynamics:** `AGE_YEARS` and `EMPLOYED_YEARS` converted from negative day offsets.
- **Affordability & Leverage Ratios:** `INCOME_PER_FAM_MEMBER` (Total Income / Family Size) and `CHILDREN_RATIO` (Children / Family Size).
- **Stability Metrics:** `EMPLOYMENT_TO_AGE_RATIO` (career stability measure) and `INCOME_TO_AGE_RATIO` (earning trajectory).
- **Asset Ownership:** `HAS_CAR_AND_REALTY` (unencumbered collateral composite).
- **Communication Reachability:** `TOTAL_CONTACTS` (sum of work phone, mobile, phone, email).
- **Peer-Group Relativity:**
  - `INCOME_TO_EDU_MEDIAN`: Applicant's income indexed to the median income of their educational cohort.
  - `TENURE_TO_OCC_MEDIAN`: Applicant's job stability indexed to the median tenure of their specific occupation group.

### 3. Decile Rank-Ordering Monotonicity

The 10-decile risk distribution confirms strict **monotonic descent of the empirical bad rate**, validating that the model reliably concentrates risk in Deciles 1–3:

- **Decile 1:** Bad Rate = **26.01%** | Cumulative Bads = 47.4%
- **Decile 2:** Bad Rate = **11.55%** | Cumulative Bads = 68.5%
- **Decile 3:** Bad Rate = **6.42%** | Cumulative Bads = 80.2% | **Peak KS = 49.82%**
- **Decile 10:** Bad Rate = **0.22%** | Cumulative Bads = 100.0%

### 4. Population Stability Index (PSI) & Drift Monitoring

$$
\text{PSI} = \sum_{i=1}^{k} (\text{Actual}\%_i - \text{Expected}\%_i) \times \ln\left(\frac{\text{Actual}\%_i}{\text{Expected}\%_i}\right)
$$

- **Overall Model PSI:** **0.0418** (Falls comfortably into the **Green Band** $< 0.10$, indicating population stability between development and production).
- **Characteristic Stability (CSI):** Key drivers (`AMT_INCOME_TOTAL`: 0.031, `DAYS_EMPLOYED`: 0.048, `DAYS_BIRTH`: 0.021) all show negligible drift.

### 4. Macroeconomic Stress Testing (Vasicek Asymptotic Credit Risk Model)

Using the single-factor Vasicek IRB formulation ($\rho = 12\%$ asset correlation):

$$
\text{PD}_{\text{stressed}} = \Phi\left(\frac{\Phi^{-1}(\text{PD}) - \sqrt{\rho} Z}{\sqrt{1 - \rho}}\right)
$$

- **Baseline Scenario:** Portfolio PD = **2.14%** | Expected Loss = $963,000 | Capital Buffer = Fully Adequate
- **Moderate Adverse Recession (+2.5% Unemployment, -1.5% GDP):** Stressed PD = **3.82%** (1.78x multiplier) | Capital Drawdown = +78.5%
- **Severely Adverse CCAR Crisis (+5.0% Unemployment, -4.0% GDP, +400bps Rate):** Stressed PD = **6.15%** (2.87x multiplier) | Capital Impact = +187.4% (Requires Countercyclical Buffer drawdown)

### 5. Fair Lending & Disparate Impact Audit (ECOA Four-Fifths Rule)

$$
\text{Disparate Impact Ratio} = \frac{\text{Approval Rate of Unprivileged Cohort}}{\text{Approval Rate of Privileged Cohort}} \ge 0.80
$$

- **Gender Disparate Impact Ratio:** **0.942** (Pass — Non-discriminatory)
- **Age Cohort Disparate Impact Ratio:** **0.887** (Pass — Non-discriminatory)

---

## 🏛️ System Architecture

```
Credit-Card-Approval-Prediction/
├── Dataset/
│   ├── application_record.csv       # Application financial and demographic attributes
│   └── credit_record.csv            # Monthly credit repayment history & status codes
├── models/
│   ├── Random_Forest_best_model.pkl # Production Champion Model
│   ├── best_threshold.txt           # Cost-calibrated policy cutoff (0.34)
│   └── train_columns.pkl            # Aligned feature schema
├── notebooks/
│   ├── 1_Visualizing_and_analyzing_data.ipynb
│   ├── 2_Data_preprocessing.ipynb
│   ├── 3_Model_building.ipynb
│   ├── 4_Prediction.ipynb
│   └── 5_Model_Risk_Evaluation_and_Validation.ipynb  # Comprehensive Validation Report
├── templates/
│   ├── landing_page.html            # Executive portal overview & KPI summary
│   ├── form.html                    # Credit application input interface with risk tags
│   ├── result.html                  # Bureau score (300-850), PD, stress PD & SHAP attributions
│   ├── model_risk_evaluation.html   # Interactive Validation Suite (Gini, KS, PSI, Stress Simulator)
│   └── audit_report.html            # Formal printable SR 11-7 / Basel III Audit Certificate
├── app.py                           # Flask web service with stress testing & validation APIs
├── model_risk_evaluation.py         # Production quantitative risk validation engine
├── Credit_Risk_and_Model_Validation_Interview_Guide.md  # Interview defense & Q&A guide
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## 🚀 How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/udityamerit/Credit-Card-Approval-Prediction.git
cd Credit-Card-Approval-Prediction
```

### 2. Set Up Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask Web Application

```bash
python app.py
```

### 5. Access the Web Application

Open your web browser and navigate to:

- **Executive Portal & Overview:** [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- **Model Risk Evaluation Dashboard:** [http://127.0.0.1:5000/model-risk-evaluation](http://127.0.0.1:5000/model-risk-evaluation)
- **Credit Scoring & PD Engine:** [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)
- **Printable SR 11-7 Audit Certificate:** [http://127.0.0.1:5000/audit-report](http://127.0.0.1:5000/audit-report)

---

## 🖥️ Web Application Features

1. **Executive Portal (`/`):** Highlights live model health, Gini (0.6934), KS (49.82%), PSI (0.0418), and regulatory compliance badges.
2. **Interactive Model Validation Suite (`/model-risk-evaluation`):**
   - **Tab 1: Discriminatory Power:** Champion vs. Challenger comparison and 10-decile rank-ordering bad rate table.
   - **Tab 2: Stability & Drift:** Population Stability Index (PSI) and Characteristic Stability Index (CSI) tracking with regulatory RAG indicators.
   - **Tab 3: Real-Time Macro Stress Testing Simulator:** Interactive dynamic sliders for unemployment, GDP contraction, and interest rate hikes that query `/api/stress-test` and update portfolio stressed PD, expected losses, and capital drawdowns in real-time.
   - **Tab 4: Model Governance & Fair Lending:** Complete US Fed SR 11-7 / OCC 2011-12 validation checklist and ECOA disparate impact ratios.
3. **Credit Application Scoring Engine (`/predict` & `/result`):**
   - Converts input features into a calibrated **Probability of Default (PD)**.
   - Computes standard **Credit Bureau Points (300–850)** and internal **Basel Risk Tiers** (AAA Prime to D Impaired).
   - Generates simulated **Stressed PD** under macroeconomic recession.
   - Provides **Explainable Risk Factor Attributions** (SHAP proxy) for adverse action compliance.
4. **Formal Audit Certificate (`/audit-report`):** Clean, print-ready formal model validation certificate for presentation to the Model Risk Governance Committee.

---


## 👨‍💻 Author & Contact

**Uditya Narayan Tiwari**

- **Portfolio:** [udityanarayantiwari.netlify.app](https://udityanarayantiwari.netlify.app/)
- **LinkedIn:** [linkedin.com/in/uditya-narayan-tiwari-562332289](https://www.linkedin.com/in/uditya-narayan-tiwari-562332289/)
- **GitHub:** [github.com/udityamerit](https://github.com/udityamerit)
- **Email:** uditmerit@gmail.com

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).
