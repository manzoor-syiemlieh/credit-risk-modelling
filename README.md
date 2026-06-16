# Credit Risk Modelling — PD, LGD, EAD & Expected Loss

An end-to-end, Basel-aligned credit risk pipeline on LendingClub loan data
(2007–2014, 466,285 loans). It builds the full expected-loss framework —
Probability of Default (PD), Loss Given Default (LGD), Exposure at Default (EAD) —
translates the PD model into an interpretable credit scorecard, and monitors model
stability with PSI.

Built by a risk professional with 7+ years at PayPal, with an emphasis on the parts
that matter in a regulated lending environment: interpretability, auditability, and
a clear line from model output to a capital decision.

---

## Why this framework

A lender doesn't just want to know *if* a borrower defaults — it needs to know how
much it expects to lose across the whole book, because that drives capital. The
Basel expected-loss identity decomposes that:

**Expected Loss = PD × LGD × EAD**

So the project builds each component, then combines them into a portfolio-level loss
figure and checks it against a capital buffer.

## Data

[LendingClub Loan Data 2007–2014](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
— 466,285 loans, 74 raw features. The raw and processed CSVs are large and aren't
committed; download the source file, place it at `data/loan_data_2007_2014.csv`, and
run notebook 01 to regenerate the processed train/test files the later notebooks need.

## Pipeline

| Notebook | What it does |
|---|---|
| `01_Data_Preprocessing_and_Preparation.ipynb` | Cleaning, Basel good/bad target, WoE/IV, fine & coarse classing, 80/20 split |
| `02_PD_Model_and_Scorecard.ipynb` | Logistic Regression PD model, evaluation, 300–850 scorecard, approval cut-offs |
| `03_PD_Monitoring_PSI.ipynb` | Population Stability Index to detect score-distribution drift |
| `04_LGD_EAD_Expected_Loss.ipynb` | Two-stage LGD, EAD via CCF, portfolio Expected Loss, capital check |

## Approach, by component

**PD — Logistic Regression on WoE features.** WoE/IV handles non-linearity and keeps
everything monotonic and interpretable; logistic regression is the regulatory
standard because its coefficients map directly to scorecard points. Reference
categories are dropped per feature group to avoid the dummy-variable trap. The model
is then rescaled into a 300–850 scorecard (FICO-style), and the scorecard is
validated by reverse-engineering PD from the score and matching it back to the model.

**LGD — two-stage model on 43,236 defaulted accounts.** Recovery rate is bounded in
[0,1] with a large spike at zero, so instead of (unavailable) Beta regression: stage 1
is a logistic model for *whether* any recovery occurs, stage 2 a linear model for
*how much*, given some recovery. LGD = 1 − (stage1 probability × stage2 rate).

**EAD — linear regression on the Credit Conversion Factor.** CCF is the share of the
funded amount drawn at default; EAD = funded_amount × predicted CCF.

**Monitoring — PSI.** Standard thresholds: < 0.1 stable, 0.1–0.25 watch, > 0.25
recalibrate.

## Results

### PD model (held-out test set, 93,257 loans)

| Metric | Value | Read |
|---|---|---|
| ROC-AUC | 0.70 | Acceptable discrimination |
| GINI | 0.40 | Above the 0.30 retail-credit threshold |
| KS | 0.30 | Good separation for an application scorecard |

These are honest, application-time numbers. The model uses only at-application
features (no behavioural or bureau-update data), so ~0.70 AUC / 0.40 GINI is in the
expected range for this kind of scorecard and clears the usual retail benchmarks —
it is not, and shouldn't be sold as, a 0.9-AUC model.

### LGD / EAD / Expected Loss

| Quantity | Value |
|---|---|
| Defaulted accounts modelled | 43,236 |
| LGD stage-1 ROC-AUC | 0.65 |
| Mean predicted LGD | ~0.92 |
| Mean predicted CCF (EAD) | ~0.74 |
| **Total Expected Loss** | **$501.6M** |
| Total funded amount | $6.66B |
| **EL as % of portfolio** | **7.52%** |

Portfolio EL of 7.52% sits below the typical 10% regulatory capital buffer, which
supports a sustainable lending posture under that constraint. The LGD stage-1 model
is the weakest link (AUC 0.65) and is the clearest target for future improvement.

## Run it

```bash
git clone https://github.com/manzoor-syiemlieh/credit-risk-modelling.git
cd credit-risk-modelling
# download loan_data_2007_2014.csv from Kaggle into data/, then run the notebooks in order 01 -> 04
```
Notebook 01 generates the processed train/test files; 02 generates the scorecard
files that 03 and 04 depend on.

## Limitations & next steps

- PD uses application-time features only; adding behavioural/bureau data would lift
  discrimination.
- LGD stage-1 separation (AUC 0.65) is modest — worth revisiting features or trying a
  gradient-boosted stage 1.
- Cost/capital thresholds are illustrative; in production they'd come from the
  lender's own loss experience and regulatory capital rules.

## Tools

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

## Author

**Manzoor Syiemlieh** — Fraud & Risk Analytics, 7+ years (PayPal) → Data Science
[LinkedIn](https://www.linkedin.com/in/manzoor-syiemlieh) ·
[GitHub](https://github.com/manzoor-syiemlieh)

MIT Licensed.
