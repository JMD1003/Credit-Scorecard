import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def tune_model(X_train, y_train):
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "saga"],
        "max_iter": [500, 1000]
    }

    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=5,
        scoring="roc_auc_ovr_weighted",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print("Best Params:", grid_search.best_params_)
    
    return grid_search.best_estimator_, grid_search.best_params_