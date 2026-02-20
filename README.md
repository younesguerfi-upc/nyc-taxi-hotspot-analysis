#  NYC Taxi Hotspot Analysis

##  Project Overview
This project identifies **spatial and temporal demand hotspots** in taxi rides in New York City using big data processing, machine learning, and interactive visualization.  
The goal is to help operators and analysts understand high-demand areas and peak times, enabling better resource allocation.

---

##  Technologies Used
- **Python 3**
- **Dask** – for large-scale data processing
- **Pandas & NumPy** – data manipulation
- **Scikit-learn & XGBoost** – machine learning models
- **Streamlit** – interactive dashboard
- **PyDeck** – heatmap visualization
- **Docker** – containerization for deployment

---

##  Project Structure
nyc-taxi-hotspot-analysis/
│
├── data/                  
│   └── preprocessed_taxi/  # processed parquet files for dashboard
│
├── model/
│   └── clf.pkl             # trained ML model
│
├── dashboard/              # Streamlit app folder
│   └── app.py              # main Streamlit dashboard
│
├── docker/                 # Docker files (optional)
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── notebooks/              # Jupyter notebooks for exploration (optional)
│
├── requirements.txt        # all Python dependencies
├── README.md               # project description
└── .gitignore              # ignore unnecessary files

---

##  Dataset

**Source:** [NYC Taxi Rides Dataset](https://huggingface.co/datasets/TaherMAfini/taxi_dataset)

**Files:**
- `train.csv` – training data with fare amounts (~55M rows)
- `test.csv` – test set for predictions (~10k rows)
- `sample_submission.csv` – example submission format

**Fields:**
- `pickup_datetime` – ride start timestamp
- `pickup_longitude`, `pickup_latitude` – pickup coordinates
- `dropoff_longitude`, `dropoff_latitude` – dropoff coordinates
- `passenger_count` – number of passengers
- `fare_amount` – target variable (training only)

---

##  Machine Learning

- Feature engineering from datetime attributes (hour, weekday, day, month)
- **Cyclical encoding** using sine and cosine for periodic features
- Labels: passenger count > 4 → binary classification
- Model: `MLPClassifier` trained on balanced samples
- Model saved with `pickle` for dashboard prediction

---

##  Dashboard

**Features:**

1. **Hotspot Prediction**
   - User selects date, hour, and location
   - Classifier predicts hotspot probability
   - Heatmap overlay on NYC map

2. **Operational Insights**
   - Aggregate rides by date, hour, zone
   - View top high-variance zones and persistent hotspots
   - Interactive charts and data tables

**Run locally:**

```bash
cd dashboard
streamlit run app.py
pip install -r requirementx.txt

**or using Docker**
docker compose up
