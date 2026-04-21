import pandas as pd
import os

def load_data(filepath: str = r"c:\Users\jaked\Downloads\german_credit_data.csv") -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file does not exist.")
    return pd.read_csv(filepath)