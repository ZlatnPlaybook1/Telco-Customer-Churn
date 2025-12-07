import os 
import pandas as pd
import mlflow


Base_DIR = os.path.dirname(os.path.abspath(__file__)) # src/serving
MODEL_DIR = os.path.join(Base_DIR, "model") # src/serving/model

model = None
try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    # try to find an mlflow model dir (a subfolder containing MLmodel) under MODEL_DIR
    print(f"Direct load from {MODEL_DIR} failed: {e}")
    for dirpath, dirnames, filenames in os.walk(MODEL_DIR):
        if "MLmodel" in filenames:
            try:
                model = mlflow.pyfunc.load_model(dirpath)
                print(f"Model loaded successfully from discovered MLflow dir: {dirpath}")
                break
            except Exception as e2:
                print(f"Failed to load model from discovered dir {dirpath}: {e2}")
    if model is None:
        print(f"Warning: model not loaded. predict() will raise if called without a model.")

try:
    # try the most-likely location first
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    if not os.path.exists(feature_file):
        # search recursively for feature_columns.txt anywhere under MODEL_DIR
        feature_file = None
        for dirpath, dirnames, filenames in os.walk(MODEL_DIR):
            if "feature_columns.txt" in filenames:
                feature_file = os.path.join(dirpath, "feature_columns.txt")
                break

    if feature_file is None:
        raise FileNotFoundError(f"feature_columns.txt not found under {MODEL_DIR}")

    with open(feature_file, "r", encoding="utf-8") as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"Loaded {len(FEATURE_COLS)} feature columns from {feature_file}")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")


# Deterministic binary feature mappings 
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},      
    "Partner": {"No": 0, "Yes": 1},              
    "Dependents": {"No": 0, "Yes": 1},              
    "PhoneService": {"No": 0, "Yes": 1},         
    "PaperlessBilling": {"No": 0, "Yes": 1},     
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    for c in  NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].fillna(0)
    
    for c , mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c].astype(str).str.strip().map(mapping).astype("Int64")
                .fillna(0).astype(int)
            )

    obj_cols = [c for c in df.select_dtypes(include=['object']).columns]
    if obj_cols:
        df = pd.get_dummies(df,
                        columns=obj_cols,
                        drop_first=True)
        
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0 :
        df[bool_cols] = df[bool_cols].astype(int)

    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df


def predict(input_dict: dict) -> str:

    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        
        if isinstance(preds, (list,tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds

    except Exception as e:
        raise Exception(f"Model predcition failed: {e}")
    

    if result == 1 :
        return "Likely to churn"
    else:
        return "Not likely to churn"

