# Time Series Anomaly Detection

Anomaly detection on synthetic sensor time series data using three methods: Z-Score statistical detection, Isolation Forest, and an LSTM prediction-based model. Includes an interactive Streamlit dashboard with adjustable parameters.

## Results

| Model | AUROC |
|---|---|
| LSTM (prediction-based) | 1.000 |
| Z-Score | 0.855 |
| Isolation Forest | 0.810 |

## Key Finding

The prediction-based LSTM achieves perfect AUROC because it learns the temporal pattern of the signal and flags points where the actual value deviates significantly from the predicted value. Z-Score and Isolation Forest treat each point independently and miss temporal context.

## Dashboard Features

- Adjustable number of data points, anomalies, and spike size
- Z-Score threshold slider
- Isolation Forest contamination slider
- LSTM sequence length slider
- Side-by-side detection plots for all three models
- AUROC metric cards updated in real time
- Model comparison bar chart

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, Streamlit, NumPy, pandas, Matplotlib

## How to Run

1. Install dependencies:
   pip install streamlit tensorflow scikit-learn numpy pandas matplotlib

2. Run the dashboard:
   streamlit run app.py

3. Adjust settings in the sidebar and click Run Detection

## Notebook

Time_Series_Anomaly_Detection.ipynb contains the full development workflow including data generation, model training, evaluation, and robustness analysis.

## Anomaly Types

- Point spikes: sudden large deviations from normal signal
- Signal: sinusoidal pattern with Gaussian noise
- Detection: unsupervised, no labels used during training
