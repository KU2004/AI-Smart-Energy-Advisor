import pandas as pd

def load_data(file_path):
    df = pd.read_csv(
        file_path,
        sep=';',
        low_memory=False,
        na_values=['?']
    )

    print("✅ Raw Data Loaded")
    return df