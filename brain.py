import pickle
import pandas as pd
from datetime import datetime

# Load model and features
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("car_features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

def predict_price(user_input: dict) -> float:
    row = {col: 0 for col in feature_columns}

    # Required fields
    row["car_age"] = datetime.now().year - user_input["year"]
    row["present_price"] = user_input["present_price"]
    row["kms_driven"] = user_input["kms_driven"]

    # One-hot fields
    for key in user_input:
        if key in row:
            row[key] = user_input[key]

    input_df = pd.DataFrame([row])[feature_columns]
    return model.predict(input_df)[0]
