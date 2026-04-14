from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"📊 RMSE: {rmse}")
    print(f"📊 R2 Score: {r2}")