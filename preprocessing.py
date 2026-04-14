import pandas as pd

def preprocess_data(df):
    # Combine date & time
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # Convert to numeric
    df['Global_active_power'] = pd.to_numeric(
        df['Global_active_power'], errors='coerce'
    )

    # Drop missing values
    df = df.dropna()

    # Rename column
    df = df.rename(columns={
        'Global_active_power': 'energy_consumption'
    })

    print("✅ Data Cleaned")

    return df[['timestamp', 'energy_consumption']]