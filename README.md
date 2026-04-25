# Credit Scorecard

A logistic regression-based credit scoring model built on the German Credit dataset. The model estimates borrower risk and converts model output into an interpretable credit score between 300–850, mirroring how banks estimate default risk under **IFRS9**.

## Key Concepts

| Concept | Description |
|---|---|
| **Weight of Evidence (WoE)** | Industry-standard encoding for credit scorecards |
| **Information Value (IV)** | Measures predictive power of each feature |
| **Gini Coefficient** | 2 × AUC − 1, primary discrimination metric |
| **KS Statistic** | Max separation between risk distributions |
| **PDO Scaling** | Converts log-odds to 300–850 credit score range |

---

## Quickstart

**1. Clone the repo and set up the virtual environment:**
```bash
git clone https://github.com/JMD1003/Credit-Scorecard.git
cd Credit-Scorecard
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r Requirements.txt
```

**2. Train the model:**
```bash
.venv\Scripts\python.exe -m SRC.Model.train
```

**3. Generate the scorecard report:**
```bash
.venv\Scripts\python.exe -m SRC.scorecard.scorecard
```

---

## Model Evaluation

| Metric | Value | Industry Benchmark |
|---|---|---|
| ROC-AUC | ~0.99 | > 0.70 acceptable |
| Gini Coefficient | ~0.99 | > 0.30 acceptable |
| KS Statistic | ~0.00 | > 0.20 acceptable |

> **Note:** The high ROC-AUC and Gini values reflect that the `Risk` target was derived from `Credit amount` using quantile thresholds, making it highly predictable by design. In a production setting, the target would be sourced from observed default outcomes.

---

## Score Bands

| Band | Score Range | Risk Level |
|---|---|---|
| A | 750 – 850 | Very Low Risk |
| B | 650 – 749 | Low Risk |
| C | 550 – 649 | Medium Risk |
| D | 300 – 549 | High Risk |

---

## Score Scaling Formula

Scores are scaled using the standard PDO (Points to Double the Odds) formula:

```python
PDO = 20        # Points to double the odds
base_score = 600
base_odds = 50

factor = PDO / log(2)
offset = base_score - (factor * log(base_odds))
score  = offset + factor * log_odds
```

---

## Tech Stack

- **Python** — pandas, numpy, scikit-learn
- **MLflow** — experiment tracking
- **joblib** — model serialization