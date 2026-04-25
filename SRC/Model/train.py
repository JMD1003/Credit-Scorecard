import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.model_selection import cross_val_score, train_test_split

from SRC.data.Load_Data import load_data
from SRC.data.Preprocess import preprocess_data
from SRC.Model.tune import tune_model
from SRC.Model.evaluate import evaluate_model

def train():
    mlflow.set_tracking_uri("file:./mlruns")

    df = load_data()
    df = preprocess_data(df)

    X = df.drop(columns=["Risk"])
    y = df["Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_model, best_params = tune_model(X_train, y_train)

    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="accuracy")
    print("CV Accuracy:", cv_scores.mean())

    metrics = evaluate_model(best_model, X_test, y_test)

    try:
        mlflow.set_tracking_uri("file:./mlruns")
        with mlflow.start_run():
            mlflow.log_params(best_params)
            mlflow.log_metrics({
                "cv_accuracy": float(cv_scores.mean()),
                "roc_auc": float(metrics["roc_auc"]),
                "gini_coefficient": float(metrics["gini"]),
                "ks_statistic": float(metrics["ks"])
            })
            mlflow.sklearn.log_model(best_model, name="credit_scorecard_model")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/credit_scorecard_model.pkl")
    print("Model saved.")
    print(f"CV Accuracy: {cv_scores.mean():.4f}")
    print(f"ROC-AUC: {float(metrics['roc_auc']):.4f}")
    print(f"Gini: {float(metrics['gini']):.4f}")
    print(f"KS: {float(metrics['ks']):.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/credit_scorecard_model.pkl")
    print("Model saved.")

if __name__ == "__main__":
    train()