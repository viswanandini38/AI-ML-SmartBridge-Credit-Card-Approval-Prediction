# importing essential libraries
from pathlib import Path
# importing joblib for model serialization
import joblib
# importing data manipulation libraries
import pandas as pd
# importing Flask and related libraries
from flask import Flask, jsonify, render_template, request
# importing machine learning libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# initializing paths and directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# defining paths for model, encoders, feature columns, and dataset
MODEL_PATH = MODELS_DIR / "card_model.joblib"
ENCODERS_PATH = MODELS_DIR / "encoders.joblib"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.joblib"
DATASET_PATH = BASE_DIR / "clean_dataset.csv"

# initializing Flask app

app = Flask(__name__, template_folder=str(BASE_DIR), static_folder=str(BASE_DIR), static_url_path="")

# training and saving the model if not already present

def train_and_save_model():
    df = pd.read_csv(DATASET_PATH)
    # Defining feature columns and target variable
    feature_cols = [
        "Gender", "Age", "Debt", "Married", "BankCustomer", "Industry",
        "YearsEmployed", "PriorDefault", "Employed", "CreditScore",
        "Citizen", "Income"
    ]

    X = df[feature_cols].copy()
    y = df["Approved"].copy()

    # Encoding categorical features using LabelEncoder
    encoders = {}
    for col in ["Gender", "Industry", "Citizen"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Saving the model, encoders, and feature columns
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)

    return model, encoders, feature_cols

# loading the model, encoders, and feature columns

def load_or_train_model():
    if not all([MODEL_PATH.exists(), ENCODERS_PATH.exists(), FEATURE_COLS_PATH.exists()]):
        print("Model files not found. Training a new model from the provided dataset...")
        return train_and_save_model()
    # If model files exist, load them
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model, encoders, feature_cols

# Load the model, encoders, and feature columns
model, encoders, feature_cols = load_or_train_model()


# ── Print mappings on startup ──
print("\n" + "=" * 50)
print("MODEL LOADED SUCCESSFULLY")
print("=" * 50)
print(f"Model    : {type(model).__name__}")
print(f"Features : {feature_cols}")
print("\nENCODER MAPPINGS:")
# encoder mappings for categorical columns
for col, le in encoders.items():
    print(f"\n  {col}:")
    for i, cls in enumerate(le.classes_):
        print(f"    '{cls}' → {i}")
print("=" * 50 + "\n")

# Utility function to safely encode categorical values

def safe_encode(col_name, value):
    """
    Encode a categorical value using the saved LabelEncoder.
    Handles string values like 'ByBirth' and numeric strings like '0' or '1'.
    """
    # Check if the column has an encoder
    le = encoders[col_name]
    classes = list(le.classes_)
    value = str(value).strip()

    if value in classes:
        return int(le.transform([value])[0])
    # If the value is a numeric string, try to convert it to an integer and check if it's a valid index
    try:
        numeric_val = int(float(value))
        if 0 <= numeric_val < len(classes):
            print(f"  ⚠️ Numeric fallback: '{value}' → {numeric_val} for {col_name}")
            return numeric_val
    except Exception:
        pass
        # If the value is not found in the classes, log a warning and return 0
    print(f"  ⚠️ Unknown value '{value}' for {col_name}. Classes: {classes}. Using 0.")
    return 0

# Flask home route 
@app.route("/")
def home():
    return render_template("index.html")

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Handling prediction requests
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")
        
        print("\n" + "=" * 50)
        print("📥 NEW PREDICTION REQUEST")
        print("=" * 50)
        print("Raw input:", data)
        # Validate required fields
        gender_enc = safe_encode("Gender", data["gender"])
        industry_enc = safe_encode("Industry", data["industry"])
        citizen_enc = safe_encode("Citizen", data["citizen"])

        print("\nEncoded:")
        print(f"  Gender   : '{data['gender']}' → {gender_enc}")
        print(f"  Industry : '{data['industry']}' → {industry_enc}")
        print(f"  Citizen  : '{data['citizen']}' → {citizen_enc}")

        # Constructing the input dictionary for prediction
        input_data = {
            "Gender": gender_enc,
            "Age": float(data["age"]),
            "Debt": float(data["debt"]),
            "Married": int(data["married"]),
            "BankCustomer": int(data["bank_customer"]),
            "Industry": industry_enc,
            "YearsEmployed": float(data["years_employed"]),
            "PriorDefault": int(data["prior_default"]),
            "Employed": int(data["employed"]),
            "CreditScore": float(data["credit_score"]),
            "Citizen": citizen_enc,
            "Income": float(data["income"]),
        }

        print(f"\nFull input dict: {input_data}")

        input_df = pd.DataFrame([input_data])[feature_cols]
        print(f"\nFeature array:\n{input_df.to_string()}")

        # Making predictions
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        confidence = round(float(max(probability)) * 100, 1)
        approved = bool(prediction == 1)

        print(f"\n🎯 Result     : {'APPROVED' if approved else 'REJECTED'}")
        print(f"Confidence : {confidence}%")
        print(f"Proba      : Rejected={probability[0]:.3f} | Approved={probability[1]:.3f}")
        print("=" * 50 + "\n")

        # Returning the prediction result as JSON
        return jsonify({"approved": approved, "confidence": confidence})

    # Handling exceptions and returning error messages
    except Exception as e:
        import traceback
        print("ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

# Running the Flask app
if __name__ == "__main__":
    app.run(debug=True)