import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.model_selection import cross_val_score, train_test_split

from SRC.data.Load_Data import load_data
from SRC.data.Preprocess import preprocess_data
from tune import tune_model
from evaluate import evaluate_model

def train():

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

    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_accuracy", cv_scores.mean())
        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        mlflow.log_metric("gini_coefficient", metrics["gini"])
        mlflow.log_metric("ks_statistic", metrics["ks"])
        mlflow.sklearn.log_model(best_model, "credit_scorecard_model")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/credit_scorecard_model.pkl")
    print("Model saved.")

if __name__ == "__main__":
    train()