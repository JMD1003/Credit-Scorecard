import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve

def gini_coefficient(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    return 2 * auc - 1

def ks_statistic(y_true, y_prob):
    y_binary = (y_true == 2).astype(int)
    y_prob_high = y_prob[:, 2]
    fpr, tpr, _ = roc_curve(y_binary, y_prob_high)
    return max(tpr - fpr)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    gini = gini_coefficient(y_test, y_prob)
    ks = ks_statistic(y_test, y_prob)

    print(f"ROC-AUC:          {roc_auc:.4f}")
    print(f"Gini Coefficient: {gini:.4f}")
    print(f"KS Statistic:     {ks:.4f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return {"roc_auc": roc_auc, "gini": gini, "ks": ks}