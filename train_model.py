import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

files = [
    "car data.csv",  # this one usually has labels
    "CAR DETAILS FROM CAR DEKHO.csv",
    "Car details v3.csv",
    "car details v4.csv"
]

usable_dfs = []
print("📥 Reading files...\n")

for file in files:
    try:
        df = pd.read_csv(file, low_memory=False)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        rename_map = {
            'name': 'car_name',
            'price': 'selling_price',
            'km_driven': 'kms_driven',
            'kilometer': 'kms_driven',
            'fuel': 'fuel_type',
            'fuel_type ': 'fuel_type',
            'seller type': 'seller_type'
        }
        df.rename(columns=rename_map, inplace=True)

        required = {'year', 'selling_price', 'present_price', 'kms_driven'}
        if not required.issubset(df.columns):
            print(f"❌ Skipped {file} — missing important columns.\n")
            continue

        before = df.shape[0]

        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
        df['present_price'] = pd.to_numeric(df['present_price'], errors='coerce')
        df['kms_driven'] = df['kms_driven'].astype(str).str.replace(",", "").str.extract(r'(\d+)')[0]
        df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')

        df = df.dropna(subset=['year', 'selling_price', 'present_price', 'kms_driven'])
        after = df.shape[0]

        if after > 0:
            usable_dfs.append(df)
            print(f"✅ {file}: {before} → {after} rows usable.\n")
        else:
            print(f"⚠️ {file}: 0 usable rows.\n")

    except Exception as e:
        print(f"❌ Failed to process {file}: {e}\n")

# ------------------------- Combine Usable Data -------------------------
if not usable_dfs:
    raise ValueError("❌ No usable data found across all files.")

df = pd.concat(usable_dfs, ignore_index=True)
print(f"📊 Final usable dataset shape: {df.shape}\n")

# ------------------------- Select & Encode -------------------------
df = df[[
    'car_name', 'year', 'selling_price', 'present_price',
    'kms_driven', 'fuel_type', 'seller_type', 'transmission', 'owner'
]]

df = pd.get_dummies(df, columns=['fuel_type', 'seller_type', 'transmission', 'owner'], drop_first=True)

df['car_age'] = datetime.now().year - df['year']
df.drop(columns=['year'], inplace=True)
df.drop(columns=['car_name'], inplace=True)

# ------------------------- Train/Test -------------------------
X = df.drop(columns=['selling_price'])
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------- Train Model -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"✅ Model trained. R² score: {r2:.4f}")

# ------------------------- Save -------------------------
with open("car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("car_features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("📦 Saved model to 'car_price_model.pkl' and features to 'car_features.pkl'")
