
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Evaluation function
def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")