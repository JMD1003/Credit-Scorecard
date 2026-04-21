import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df.columns = df.columns.str.strip()
    
    for col in ["Unnamed: 0"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    if "Sex" in df.columns and df["Sex"].dtype == "object":
        df["Sex"] = df["Sex"].str.strip().map({"male": 0, "female": 1})
        
    if "Housing" in df.columns and df["Housing"].dtype == "object":
        df = pd.get_dummies(df, columns = ["Housing"], drop_first=False)
        
    if "Saving accounts" in df.columns and df["Saving accounts"].dtype == "object":
        df = pd.get_dummies(df, columns = ["Saving accounts"], drop_first=False)
        
    if "Checking account" in df.columns and df["Checking account"].dtype == "object":
        df = pd.get_dummies(df, columns = ["Checking account"], drop_first=False)
        
    if "Purpose" in df.columns and df["Purpose"].dtype == "object":
        df = pd.get_dummies(df, columns = ["Purpose"], drop_first=False)
    
    df["Risk"] = pd.cut(df["Credit amount"], 
                    bins = [-float("inf"),
                            df["Credit amount"].quantile(0.33),
                            df["Credit amount"].quantile(0.66),
                            float("inf")], 
                    labels = ["Low", "Medium", "High"])
    
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    return df