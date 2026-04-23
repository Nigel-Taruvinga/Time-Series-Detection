
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Time Series Anomaly Detection", layout="wide")
st.title("Time Series Anomaly Detection Dashboard")
st.markdown("Real-time anomaly detection using Z-Score, Isolation Forest, and LSTM on sensor data.")

# Sidebar controls
st.sidebar.header("Settings")
n_points     = st.sidebar.slider("Number of data points", 500, 2000, 1000, step=100)
n_anomalies  = st.sidebar.slider("Number of anomalies", 10, 80, 30, step=5)
spike_size   = st.sidebar.slider("Anomaly spike size", 8, 25, 15, step=1)
zscore_thresh = st.sidebar.slider("Z-Score threshold", 1.0, 5.0, 3.0, step=0.1)
iso_contam   = st.sidebar.slider("Isolation Forest contamination", 0.01, 0.1, 0.025, step=0.005)
seq_len      = st.sidebar.slider("LSTM sequence length", 5, 30, 20, step=5)
run_btn      = st.sidebar.button("Run Detection")

if run_btn:
    with st.spinner("Generating data and running detection..."):

        # Generate data
        np.random.seed(42)
        time_steps = np.arange(n_points)
        signal = (
            10 * np.sin(2 * np.pi * time_steps / 50) +
            np.random.normal(0, 0.3, n_points)
        )
        labels = np.zeros(n_points, dtype=int)
        anomaly_indices = np.sort(np.random.choice(
            np.arange(100, n_points - 100), size=n_anomalies, replace=False
        ))
        for idx in anomaly_indices:
            signal[idx] += np.random.choice([-1, 1]) * np.random.uniform(spike_size * 0.8, spike_size * 1.2)
            labels[idx] = 1

        df = pd.DataFrame({"timestamp": time_steps, "value": signal, "anomaly": labels})

        # Plot raw signal
        st.subheader("Sensor Signal with Injected Anomalies")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df["timestamp"], df["value"], color="#7bb8e0", linewidth=0.8, label="Signal")
        ax.scatter(df[df["anomaly"]==1]["timestamp"], df[df["anomaly"]==1]["value"],
                   color="red", s=40, zorder=5, label="True anomalies")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Sensor Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Z-Score
        mean, std = signal.mean(), signal.std()
        zscore_scores = np.abs((signal - mean) / std)
        zscore_preds  = (zscore_scores > zscore_thresh).astype(int)
        zscore_auroc  = roc_auc_score(labels, zscore_scores)

        # Isolation Forest
        scaler = MinMaxScaler()
        values_scaled = scaler.fit_transform(df[["value"]])
        iso = IsolationForest(n_estimators=100, contamination=iso_contam, random_state=42)
        iso.fit(values_scaled)
        iso_scores = -iso.score_samples(values_scaled)
        iso_preds  = (iso.predict(values_scaled) == -1).astype(int)
        iso_auroc  = roc_auc_score(labels, iso_scores)

        # LSTM
        def create_pred_sequences(data, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i+seq_len])
                y.append(data[i+seq_len])
            return np.array(X), np.array(y)

        normal_values = scaler.transform(df[df["anomaly"]==0][["value"]].values)
        X_train, y_train = create_pred_sequences(normal_values, seq_len)
        X_full, y_full   = create_pred_sequences(values_scaled, seq_len)

        lstm_model = Sequential([
            LSTM(64, input_shape=(seq_len, 1), return_sequences=True),
            LSTM(32, return_sequences=False),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        lstm_model.compile(optimizer="adam", loss="mse")
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        lstm_model.fit(X_train, y_train, epochs=30, batch_size=32,
                       validation_split=0.1, callbacks=[early_stop], verbose=0)

        y_pred = lstm_model.predict(X_full, verbose=0).flatten()
        lstm_errors = np.abs(y_pred - y_full.flatten())
        lstm_labels = labels[seq_len:]
        lstm_auroc  = roc_auc_score(lstm_labels, lstm_errors)
        threshold   = np.percentile(lstm_errors, 95)
        lstm_preds  = (lstm_errors > threshold).astype(int)

        # Metrics
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Z-Score AUROC", f"{zscore_auroc:.4f}")
        col2.metric("Isolation Forest AUROC", f"{iso_auroc:.4f}")
        col3.metric("LSTM AUROC", f"{lstm_auroc:.4f}")

        # Detection plots
        st.subheader("Detection Results")
        fig2, axes = plt.subplots(3, 1, figsize=(14, 12))

        for ax, scores, preds, title, color in zip(
            axes,
            [zscore_scores, iso_scores, lstm_errors],
            [zscore_preds, iso_preds, lstm_preds],
            ["Z-Score", "Isolation Forest", "LSTM Prediction Error"],
            ["#7bb8e0", "#7bcea0", "#e07b7b"]
        ):
            ts = df["timestamp"].values if title != "LSTM Prediction Error" else df["timestamp"].values[seq_len:]
            lbl = labels if title != "LSTM Prediction Error" else lstm_labels
            ax.plot(ts, scores, color=color, linewidth=0.8, label=title)
            ax.scatter(ts[lbl==1], scores[lbl==1], color="red", s=30, zorder=5, label="True anomalies")
            ax.set_title(f"{title} (AUROC: {roc_auc_score(lbl, scores):.4f})")
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Timestamp")
        plt.tight_layout()
        st.pyplot(fig2)

        # Comparison bar chart
        st.subheader("AUROC Comparison")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        models = ["Z-Score", "Isolation Forest", "LSTM"]
        aurocs = [zscore_auroc, iso_auroc, lstm_auroc]
        colors = ["#7bb8e0", "#7bcea0", "#e07b7b"]
        bars = ax3.bar(models, aurocs, color=colors)
        ax3.set_ylim(0.5, 1.05)
        ax3.set_ylabel("AUROC")
        ax3.set_title("Model Comparison")
        for bar, val in zip(bars, aurocs):
            ax3.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                     f"{val:.4f}", ha="center", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig3)

        st.success("Detection complete!")

else:
    st.info("Adjust settings in the sidebar and click Run Detection to start.")
