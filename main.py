from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.model import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_results

from sklearn.model_selection import train_test_split

def main():
    # Load data
    df = load_data("data/household_power_consumption.txt")

    # Preprocess
    df = preprocess_data(df)

    # 🔥 IMPORTANT: reduce size (dataset is huge)
    df = df.sample(50000)

    # Feature engineering
    df = create_features(df)

    # Features & target
    X = df[['hour', 'day', 'month']]
    y = df['energy_consumption']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train model
    model = train_model(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    evaluate_model(y_test, y_pred)

    # Visualize
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()