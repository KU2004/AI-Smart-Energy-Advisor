from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=50)

    model.fit(X_train, y_train)

    # Create folder if not exists
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.pkl")

    print("✅ Model Trained & Saved")
    return model