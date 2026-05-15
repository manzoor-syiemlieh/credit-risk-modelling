# Credit Risk Modelling — PD, LGD, EAD & Scorecard Development

## Project Overview
End-to-end Basel-aligned credit risk pipeline built on real-world 
LendingClub loan data (2007–2014) covering 466,285 loan records. 
Models the full expected loss framework — Probability of Default (PD), 
Loss Given Default (LGD), Exposure at Default (EAD) — and translates 
PD outputs into an interpretable credit scorecard for lending decisions.

## Business Objective
- Assess borrower default risk using application-time features
- Build an interpretable credit scorecard for lending decisions
- Estimate potential loss exposure through LGD and EAD modelling
- Calculate Expected Loss to support capital allocation decisions
- Monitor model stability on new population data using PSI

## Project Structure
| Notebook | Description |
|----------|-------------|
| 01_Data_Preprocessing_and_Preparation.ipynb | Data cleaning, target variable creation, WoE/IV, Fine & Coarse Classing |
| 02_PD_Model_and_Scorecard.ipynb | Logistic Regression PD model, scorecard development, model evaluation |
| 03_PD_Monitoring_PSI.ipynb | Population Stability Index monitoring on new data |
| 04_LGD_EAD_Expected_Loss.ipynb | LGD and EAD modelling, Expected Loss calculation |

## Technical Approach

### Data Preparation
- Loaded and cleaned 466,285 LendingClub loan records (2007–2014)
- Engineered features from raw date and categorical variables
- Defined binary target variable using Basel II good/bad convention
- Applied 80/20 train/test split before WoE transformation
- Fine Classing (50 bins) and Coarse Classing based on WoE monotonicity
- Selected features using Information Value (IV) thresholds

### PD Model
- Logistic Regression on WoE-transformed features
- Evaluated using ROC-AUC, GINI coefficient and KS Statistic
- Translated model outputs into interpretable credit scorecard
- Scorecard enables transparent, rule-based lending decisions

### Model Monitoring
- Population Stability Index (PSI) to detect score distribution shifts
- PSI < 0.1: No significant change
- PSI 0.1–0.25: Moderate shift — monitor closely
- PSI > 0.25: Significant shift — model recalibration required

### LGD and EAD Models
- LGD: Two-stage model (Logistic + Linear Regression)
- EAD: Logistic Regression on credit conversion factors
- Expected Loss = PD × LGD × EAD
- Capital decision based on 10% buffer constraint

## Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Key Results
- PD model achieves strong discriminatory power validated by 
  ROC-AUC, GINI and KS Statistic
- Credit scorecard provides transparent lending decision framework
- Expected Loss analysis supports balanced lending posture under 
  10% capital buffer constraint
- PSI monitoring confirms model stability on new population data

## Dataset & Data Files

### Original Dataset
LendingClub Loan Data 2007–2014
- 466,285 loan records, 74 raw features
- Download from Kaggle:
  https://www.kaggle.com/datasets/wordsforthewise/lending-club
- Place as: data/loan_data_2007_2014.csv

### Generated Data Files
Run 01_Data_Preprocessing_and_Preparation.ipynb first to generate:
- data/loan_data_inputs_train.csv (373,028 records)
- data/loan_data_targets_train.csv
- data/loan_data_inputs_test.csv (93,257 records)
- data/loan_data_targets_test.csv

These files are required to run 02_PD_Model_and_Scorecard.ipynb

### Scorecard Files
Running 02_PD_Model_and_Scorecard.ipynb generates:
- data/inputs_train_with_ref_cat.csv
- data/df_scorecard.csv

These files are required for subsequent notebooks.

## Author
**Manzoor Syiemlieh**
Data Scientist | Fraud & Risk Analytics | 7+ Years Fintech @ PayPal
[LinkedIn](https://www.linkedin.com/in/manzoor-syiemlieh)
