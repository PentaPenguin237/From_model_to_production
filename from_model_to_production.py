import os
import pandas as pd
import numpy as np
import warnings
import joblib
import shutil
from sklearn.ensemble import IsolationForest
import kagglehub
import time

# --- Configuration ---
# Paths mapped via Docker Compose
DATA_DIR = "/data"
RESULTS_DIR = "/results"
DATA_FILE = "predictive_maintenance.csv"
FILE_PATH = os.path.join(DATA_DIR, DATA_FILE)
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "isolation_forest_model.joblib")

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore')

def load_data():
    """
    Loads data from the mounted volume. 
    If not found, downloads it via KaggleHub and saves it permanently to the volume.
    """
    print(f"--- [1/5] Loading Data ---")
    
    # Check if the file already exists in our permanent storage
    if not os.path.exists(FILE_PATH):
        print(f"File not found at {FILE_PATH}.")
        print("Initiating automated download via KaggleHub...")
        
        try:
            # 1. Download to temporary cache
            # This downloads the latest version of the dataset
            cached_path = kagglehub.dataset_download("shivamb/machine-predictive-maintenance-classification")
            print(f"Download complete. Cached at: {cached_path}")
            
            # 2. Locate the specific CSV file in the download
            downloaded_file = os.path.join(cached_path, DATA_FILE)
            
            if not os.path.exists(downloaded_file):
                 # Fallback search if the filename is slightly different
                 files = os.listdir(cached_path)
                 csv_files = [f for f in files if f.endswith('.csv')]
                 if csv_files:
                     downloaded_file = os.path.join(cached_path, csv_files[0])
                 else:
                     raise FileNotFoundError("No CSV file found in the downloaded dataset.")

            # 3. Move the file to our persistent /data volume
            print(f"Moving file to permanent storage: {FILE_PATH}")
            shutil.move(downloaded_file, FILE_PATH)
            print("File saved permanently.")

        except Exception as e:
            print(f"CRITICAL ERROR: Automatic download failed: {e}")
            return None

    # Load the file (whether it was just downloaded or was already there)
    print(f"Reading data from: {FILE_PATH}")
    try:
        df = pd.read_csv(FILE_PATH)
        print("Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def engineer_features(df):
    print("\n--- [2/5] Feature Engineering ---")
    
    # 1. Select the raw columns we need
    features_df = df[['Air temperature [K]', 'Rotational speed [rpm]']].copy()
    
    # 2. Rename the Rotational Speed
    features_df.rename(columns={
        'Rotational speed [rpm]': 'sound_volume_proxy'
    }, inplace=True)

    # 3. Create the Celsius column from the Kelvin column
    # Explicitly creating a new column avoids "lying variables"
    features_df['temperature_celsius'] = features_df['Air temperature [K]'] - 273.15

    # DEBUG PRINT: Verify the conversion
    print("\n[DEBUG] Checking Celsius Conversion (First 5):")
    print(features_df[['Air temperature [K]', 'temperature_celsius']].head(5).to_string())

    # 4. Drop the Kelvin column safely
    features_df.drop(columns=['Air temperature [K]'], inplace=True)

    # 5. Generate synthetic 'humidity'
    print("Generating synthetic humidity feature...")
    base_humidity = 45.0
    temp_mean = features_df['temperature_celsius'].mean()
    rpm_mean = features_df['sound_volume_proxy'].mean()

    temp_weight = -0.5
    rpm_weight = 0.005 

    features_df['humidity'] = (
        base_humidity + \
        (features_df['temperature_celsius'] - temp_mean) * temp_weight + \
        (features_df['sound_volume_proxy'] - rpm_mean) * rpm_weight
    )

    # Add noise & Clip
    noise = np.random.uniform(-2.5, 2.5, size=len(features_df))
    features_df['humidity'] += noise
    features_df['humidity'] = features_df['humidity'].clip(0, 100)
    
    # Re-order columns explicitly to ensure consistency for training
    features_df = features_df[['sound_volume_proxy', 'temperature_celsius', 'humidity']]

    print("Feature engineering complete.")
    print(features_df.describe().to_string()) 
    return features_df

def train_model(features_df):
    print("\n--- [3/5] Training Isolation Forest ---")
    # Save the feature names to ensure alignment during prediction (conceptually)
    print(f"Training on features: {list(features_df.columns)}")
    
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(features_df)
    print("Model training complete.")
    return model

def save_model(model):
    print("\n--- [4/5] Saving Model ---")
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")

def run_simulation(model):
    print("\n--- [5/5] Running Real-Time Simulation ---")
    
    # Simulation Data Stream 
    simulated_stream = [
        {
            "description": "Normal Operation (Start of Shift)",
            "data": {"temperature_celsius": 26.0, "humidity": 45.0, "sound_volume_proxy": 1500}
        },
        {
            "description": "Normal Operation (Mid-Shift)",
            "data": {"temperature_celsius": 27.5, "humidity": 42.1, "sound_volume_proxy": 1550}
        },
        {
            "description": "Anomalous Scenario (High Vibration/Temp)",
            "data": {"temperature_celsius": 35.0, "humidity": 30.0, "sound_volume_proxy": 2800}
        },
        {
            "description": "Normal Operation (After Adjustment)",
            "data": {"temperature_celsius": 26.5, "humidity": 44.0, "sound_volume_proxy": 1510}
        }
    ]

    for item in simulated_stream:
        print(f"\n[STREAM] New sensor data: {item['description']}")
        
        # Prepare data for prediction
        data = item['data']
        
        input_df = pd.DataFrame([[
            data['sound_volume_proxy'], 
            data['temperature_celsius'], 
            data['humidity']
        ]], columns=['sound_volume_proxy', 'temperature_celsius', 'humidity'])

        try:
            prediction = model.predict(input_df)
            anomaly_score = model.decision_function(input_df)
            is_anomaly = True if prediction[0] == -1 else False
            
            score_val = round(anomaly_score[0], 4)

            if is_anomaly:
                print(f"[RESULT] ANOMALY DETECTED (Score: {score_val}) - Alert sent!")
            else:
                print(f"[RESULT] Normal Operation (Score: {score_val})")
                
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            # Print expected features for debugging
            if hasattr(model, "feature_names_in_"):
                print(f"Model expects: {model.feature_names_in_}")

        time.sleep(1) 

def main():
    # 1. Load (and download if missing)
    df = load_data()
    if df is None:
        return

    # 2. Engineer
    features_df = engineer_features(df)

    # 3. Train
    model = train_model(features_df)

    # 4. Save
    save_model(model)

    # 5. Simulate
    run_simulation(model)
    
    print("\n--- Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()