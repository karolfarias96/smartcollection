import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score
import os

# Visual Configuration
sns.set_style("whitegrid")

# --- STEP 1: Data Acquisition ---
URL_SANTANDER = "http://datos.santander.es/rest/datasets/residuos_contenedores.json"
LOCAL_FILENAME = "waste_containers.json"
df_infrastructure = None

print("--- 1. Starting Infrastructure Data Acquisition ---")

# Headers to mimic a browser and avoid 403 Forbidden errors
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

try:
    # Attempt A: Local File (Faster)
    if os.path.exists(LOCAL_FILENAME):
        print(f"Reading local file '{LOCAL_FILENAME}'...")
        import json
        with open(LOCAL_FILENAME, 'r', encoding='utf-8') as f:
            content = json.load(f)
        # Handle Santander JSON structure variations
        if 'resources' in content: df_raw = pd.DataFrame(content['resources'])
        elif 'items' in content: df_raw = pd.DataFrame(content['items'])
        else: df_raw = pd.DataFrame(content)
        
    # Attempt B: Direct Download
    else:
        print(f"Attempting to download from: {URL_SANTANDER} ...")
        response = requests.get(URL_SANTANDER, headers=headers, timeout=20)
        data_json = response.json()
        if 'resources' in data_json: df_raw = pd.DataFrame(data_json['resources'])
        elif 'items' in data_json: df_raw = pd.DataFrame(data_json['items'])
        else: df_raw = pd.json_normalize(data_json)
        print("Data downloaded successfully via Internet!")

    # Cleaning and Renaming Columns
    # Removing prefixes like 'ayto:' or 'dc:'
    df_raw.columns = [col.split(':')[-1] for col in df_raw.columns]
    
    cols_map = {
        'identifier': 'container_id', 
        'latitud': 'latitude', 
        'longitud': 'longitude', 
        'residuo': 'waste_type',
        'capacidad': 'capacity'
    }
    df_infrastructure = df_raw.rename(columns=cols_map)
    
    # Convert Lat/Lon to numeric
    for col in ['latitude', 'longitude']:
        df_infrastructure[col] = pd.to_numeric(df_infrastructure[col], errors='coerce')
    
    # Filter valid locations and relevant waste types
    df_infrastructure = df_infrastructure.dropna(subset=['latitude', 'longitude'])
    relevant_types = ['Envases', 'Organico', 'Resto']
    df_infrastructure = df_infrastructure[df_infrastructure['waste_type'].isin(relevant_types)]
    
    # Limit to 600 sensors for simulation performance (Representative Sample)
    df_infrastructure = df_infrastructure.head(600)
    
    print(f"Infrastructure Ready: {len(df_infrastructure)} active sensors processed.")

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    print("Please ensure the JSON file is in the folder or check your internet connection.")
    exit()

# --- STEP 2: Historical Simulation (Crisis Scenarios) ---
print("\n--- 2. Generating Historical Data (Scenario: Heavy Rain & Floods) ---")

DAYS = 90 
# Using 'h' to avoid FutureWarning
timestamps = pd.date_range(end=datetime.now(), periods=DAYS*24, freq='h') 
history_data = []

for idx, row in df_infrastructure.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    # Random initial state
    current_fill = np.random.randint(0, 50)
    
    for ts in timestamps:
        # --- Weather Simulation ---
        is_day = 8 <= ts.hour <= 20
        rain_prob = 0.15 # 15% chance of rain
        precipitation = 0
        
        if np.random.rand() < rain_prob:
            precipitation = np.random.exponential(8) # mm/h (Heavy rain possibility)
        
        # Temperature affected by rain and time
        temp = 20 + (5 if is_day else 0) - (0.5 * precipitation) + np.random.normal(0, 1)
        humidity = 60 + (30 if precipitation > 0 else 0) + np.random.normal(0, 2)
        
        # --- Waste Dynamics ---
        base_rate = 2.0 if is_day else 0.5
        
        # HYPOTHESIS: Extreme Weather Event (>10mm)
        # Simulates debris runoff, wet waste expansion, or blocked access
        extra_load = 0
        if precipitation > 10:
            extra_load = 30 # Sudden surge (Crisis event)
        elif precipitation > 2:
            base_rate *= 1.5 # Moderate rain impact
            
        trash_in = np.random.poisson(base_rate) + extra_load
        current_fill += trash_in
        
        # Collection Logic
        if current_fill >= 90: 
            current_fill = 0 # Emptying
        elif ts.hour == 4 and np.random.rand() > 0.95:
            current_fill = 0 # Scheduled collection
            
        history_data.append({
            'latitude': lat,
            'longitude': lon,
            'hour': ts.hour,
            'day_of_week': ts.dayofweek,
            'temperature': temp,
            'humidity': humidity,
            'precipitation': precipitation,
            'fill_level': min(current_fill, 100) # Cap at 100%
        })

df_full = pd.DataFrame(history_data)

# Feature Engineering: Accumulated Rain (Trigger for floods)
df_full['rain_accumulated_3h'] = df_full['precipitation'].rolling(3).sum().fillna(0)

# Target Definition (> 85% Full)
df_full['target'] = (df_full['fill_level'] > 85).astype(int)

print(f"Dataset Generated: {len(df_full)} records.")
print(f"Critical Event Ratio: {df_full['target'].mean():.4f}")

# --- STEP 3: Modeling (High Sensitivity Strategy) ---
print("\n--- 3. Training Model (Random Forest with 1:1 Balancing) ---")

features = ['latitude', 'longitude', 'hour', 'day_of_week', 
            'temperature', 'humidity', 'precipitation', 'rain_accumulated_3h']

X = df_full[features]
y = df_full['target']

# 1. Split Data (Maintain original distribution in Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. Apply Rigid Balancing (Undersampling) on Training Set ONLY
print("Applying Strategic Undersampling...")
train_df = X_train.copy()
train_df['target'] = y_train

class_0 = train_df[train_df['target'] == 0]
class_1 = train_df[train_df['target'] == 1]

# Force 1:1 Ratio (To prioritize Recall/Safety)
class_0_under = class_0.sample(len(class_1), random_state=42)
train_balanced = pd.concat([class_0_under, class_1])

X_train_bal = train_balanced[features]
y_train_bal = train_balanced['target']

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# Train Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train_bal)

# Predict
y_pred = rf_model.predict(X_test_scaled)

# --- STEP 4: Results & Visualization ---
print("\n--- Final Results ---")
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(f"Recall (Critical Event Detection): {rec:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 1. Feature Importance Plot
plt.figure(figsize=(10, 6))
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=importances, y=importances.index, hue=importances.index, palette='viridis', legend=False)
plt.title("Feature Importance (Focus on Climate Variables)")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
print("Plot saved: 'feature_importance.png'")

# 2. Confusion Matrix Plot
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
print("Plot saved: 'confusion_matrix.png'")

# Save Metrics to Text
with open("results_metrics.txt", "w") as f:
    f.write("--- RANDOM FOREST RESULTS ---\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Recall (Sensitivity): {rec:.4f}\n\n")
    f.write(classification_report(y_test, y_pred))
print("Metrics saved to 'results_metrics.txt'")
