# 🚛 SmartCollection

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Model-Random_Forest-green)
![Status](https://img.shields.io/badge/Status-Research_Prototype-orange)

**SmartCollection** is a predictive model that integrates **IoT waste container data** with **weather conditions** to anticipate critical overflow events and prevent urban flooding.

## 📌 Project Overview
This project validates the hypothesis that fusing **climatic variables** (precipitation, temperature) with **operational sensor data** significantly improves the detection of critical waste levels compared to traditional reactive methods.

We utilized a **hybrid data approach**:
1.  **Real Infrastructure:** Geolocations and capacities from Santander Open Data (Spain).
2.  **Simulation:** Stochastic history generation to model the impact of heavy rain on waste accumulation.

## 🚀 Key Features
* **Multimodal Data Fusion:** Combines location, time, and weather metrics.
* **High Sensitivity Strategy:** Uses a **Random Forest** model with rigid balancing (1:1) to prioritize **Recall**, ensuring critical overflow events are detected before they cause drainage blockage.
* **Automated Pipeline:** Fetches data, simulates scenarios, trains the model, and visualizes results.

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/karolfarias96/smartcollection.git](https://github.com/karolfarias96/smartcollection.git)
    cd smartcollection
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn requests
    ```

3.  **Run the simulation:**
    ```bash
    python main.py
    ```

## 📊 Results
The model achieved a **Recall of ~53%** for critical events in test scenarios, validating that **Precipitation** is a top predictor for proactive waste management logistics during climate crises.

---
*Developed for academic research on Smart Cities and Disaster Prevention.*
