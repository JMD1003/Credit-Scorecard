import numpy as np
import pandas as pd
import joblib

def load_model(filepath: str = "models/credit_scorecard_model.pkl"):
    model = joblib.load(filepath)
    return model

def calculate_score(model, X):

    PDO = 20          
    base_score = 600
    base_odds = 50

    factor = PDO / np.log(2)
    offset = base_score - (factor * np.log(base_odds))

    log_odds = model.predict_log_proba(X)[:, 0]  
    scores = offset + factor * log_odds

    scores = np.clip(scores, 300, 850)
    return scores

def assign_risk_band(score):
    if score >= 750:
        return "A - Very Low Risk"
    elif score >= 650:
        return "B - Low Risk"
    elif score >= 550:
        return "C - Medium Risk"
    else:
        return "D - High Risk"

def generate_report(X, scores):
    report = X.copy()
    report["Credit Score"] = scores.astype(int)
    report["Risk Band"] = report["Credit Score"].apply(assign_risk_band)
    return report

def save_report(report, filepath: str = "outputs/scorecard_report.csv"):
    import os
    os.makedirs("outputs", exist_ok=True)
    report.to_csv(filepath, index=False)
    print(f"Report saved to {filepath}")

def run_scorecard(X):
    model = load_model()
    scores = calculate_score(model, X)
    report = generate_report(X, scores)
    save_report(report)
    print(report[["Credit Score", "Risk Band"]].head(10))
    return report

if __name__ == "__main__":
    from SRC.data.Load_Data import load_data
    from SRC.data.Preprocess import preprocess_data

    df = load_data()
    df = preprocess_data(df)
    X = df.drop(columns=["Risk"])

    print("Running scorecard...")
    run_scorecard(X)
    print("Done.")