#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-10T09:21:52.300Z
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import io
import zipfile
import traceback

st.set_page_config(page_title="Data Analyser", layout="wide")

# -------------------------
# Initialize session state
# -------------------------

if "plot_buf" not in st.session_state:
    st.session_state["plot_buf"] = None        # last generated plot (BytesIO)
if "pred_buf" not in st.session_state:
    st.session_state["pred_buf"] = None        # predictions file buffer (BytesIO)
if "pred_df" not in st.session_state:
    st.session_state["pred_df"] = None         # actual pandas DataFrame of preds
if "ml_plot_buf" not in st.session_state:
    st.session_state["ml_plot_buf"] = None     # model-eval plot (BytesIO)
if "y_test" not in st.session_state:
    st.session_state["y_test"] = None
if "y_pred" not in st.session_state:
    st.session_state["y_pred"] = None

# small helper
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# -------------------------
# App header
# -------------------------
st.title("📊 Data Analyser")

# -------------------------
# File uploader & preview
# -------------------------
uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error("Failed to read file: " + str(e))
        st.stop()
else:
    st.info("Upload a file to begin.")
    st.stop()

st.info(f" Dataset Shape Rows: {df.shape[0]} | Columns: {df.shape[1]}")


# Data preview & stats

st.subheader("View Specific Columns")
selected_cols = st.multiselect(
    "Select columns to view",
    options=df.columns.tolist())
if selected_cols:
    st.dataframe(df[selected_cols])

st.subheader("View Specific Rows")
start_row = st.number_input("Start row index", min_value=0, max_value=len(df)-1, value=0)
end_row = st.number_input("End row index", min_value=0, max_value=len(df)-1, value=min(10, len(df)-1))

if start_row <= end_row:
    st.dataframe(df.iloc[start_row:end_row+1])


    st.subheader("📊 Exploratory Data Analysis (EDA)")
    st.dataframe(df.describe())
    
st.subheader("📈 Advanced Statistics")
col1, col2 = st.columns(2)
with col1:
    st.write("*Missing Values:*")
    st.write(df.isnull().sum())
with col2:
    st.write("*Data Types:*")
    st.write(df.dtypes)

st.write("*Correlation Matrix (numeric):*")
st.dataframe(df.corr(numeric_only=True))

# -------------------------
# Visualization section
# -------------------------
st.subheader("📉 Visualization")
cols = df.columns.tolist()
x_col = st.selectbox("X-axis", options=cols, index=0)
y_col = st.selectbox("Y-axis", options=cols, index=1 if len(cols) > 1 else 0)
graph_type = st.selectbox("Select Graph Type", ["Line", "Bar", "Scatter", "Histogram", "Boxplot", "Heatmap", "Pairplot"])

if st.button("Generate Graph"):
    try:
        # Make a fresh figure for each graph type then save to session_state
        if graph_type == "Line":
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
        elif graph_type == "Bar":
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
        elif graph_type == "Scatter":
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        elif graph_type == "Histogram":
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[x_col].dropna(), kde=True, ax=ax)
        elif graph_type == "Boxplot":
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
        elif graph_type == "Heatmap":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax, cmap="coolwarm")
        elif graph_type == "Pairplot":
            # pairplot returns a PairGrid; get the underlying figure to save
            pg = sns.pairplot(df.select_dtypes(include=[np.number]).dropna().iloc[:, :6])  # limit to first 6 numeric cols
            fig = pg.fig
        # Save to buffer and show
        buf = fig_to_bytes(fig)
        st.session_state["plot_buf"] = buf
        st.image(buf)
        st.success("Visualization generated and saved.")
    except Exception as e:
        st.error("Visualization error: " + str(e))
        st.text(traceback.format_exc())

# -------------------------
# Machine learning section
# -------------------------
st.subheader("🤖 Machine Learning Prediction")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.warning("Not enough numeric columns for prediction (need at least 2 numeric columns).")
else:
    target_col = st.selectbox("Select Target (what you want to predict)", options=numeric_cols, index=0)
    feature_cols = st.multiselect("Select Features (independent variables)", options=[c for c in numeric_cols if c != target_col])

    if feature_cols:
        if st.button("Train & Predict"):
            try:
                X = df[feature_cols].dropna()
                y = df.loc[X.index, target_col]

                # split + train
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # store cleaned series in session_state (reset index so alignment when plotting)
                st.session_state["y_test"] = pd.Series(y_test).reset_index(drop=True)
                st.session_state["y_pred"] = pd.Series(y_pred).reset_index(drop=True)

                # create predictions dataframe and store
                pred_df = pd.DataFrame({"Actual": st.session_state["y_test"], "Predicted": st.session_state["y_pred"]})
                st.session_state["pred_df"] = pred_df

                # Save predictions into a bytes buffer (try Excel then CSV fallback)
                pred_buf = io.BytesIO()
                try:
                    # Try Excel (requires openpyxl)
                    with pd.ExcelWriter(pred_buf, engine="openpyxl") as writer:
                        pred_df.to_excel(writer, index=False, sheet_name="predictions")
                    pred_buf.seek(0)
                except Exception:
                    # fallback to CSV
                    pred_buf = io.BytesIO(pred_df.to_csv(index=False).encode("utf-8"))

                st.session_state["pred_buf"] = pred_buf

                # create evaluation scatter plot (actual vs predicted)
                fig_eval, ax_eval = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x=st.session_state["y_test"], y=st.session_state["y_pred"], ax=ax_eval)
                ax_eval.set_xlabel("Actual")
                ax_eval.set_ylabel("Predicted")
                ax_eval.set_title("Model Evaluation")
                eval_buf = fig_to_bytes(fig_eval)
                st.session_state["ml_plot_buf"] = eval_buf

                # Show results and metrics
                st.success("Model Trained Successfully!")
                st.subheader("### Model Evaluation:")
                if st.session_state.get("ml_plot_buf") is not None:
                    st.image(st.session_state["ml_plot_buf"], caption="Model Evaluation Plot")
                else:
                    st.info("No Evaluation plot available yet.")
                st.write(f"R² Score: {r2_score(y_test, y_pred):.3f}")
                st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}")
                st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}")
                st.dataframe(pred_df) 
            except Exception as e:
                st.error("ML Error: " + str(e))
                st.text(traceback.format_exc())
    else:
        st.info("Select at least one feature column before training.")

# -------------------------
# Saved visuals & predictions display
# -------------------------
show_saved_viz = st.checkbox("Show Saved Visualization")

if show_saved_viz:
    st.subheader("📌 Saved Visualization")
    if st.session_state.get("plot_buf") is not None:
        st.image(st.session_state["plot_buf"])

show_saved_pred = st.checkbox("Show Saved ML Prediction")

if show_saved_pred:
    st.subheader("📌 Saved ML Prediction")
    if st.session_state.get("pred_df") is not None:
        st.dataframe(st.session_state["pred_df"])

show_saved_eval = st.checkbox("Show Model Evaluation")

if show_saved_eval:
    st.subheader("📌 Saved Model Evaluation")
    if st.session_state.get("ml_plot_buf") is not None:
        st.image(st.session_state["ml_plot_buf"])


# -------------------------
# Download selected items (ZIP)
# -------------------------
st.subheader("📥 Download Selected Items")
include_clean = st.checkbox("Cleaned Dataset (CSV)", value=True)
include_corr = st.checkbox("Correlation Matrix (CSV)", value=True)
include_plots = st.checkbox("Visualisation Plot (PNG)", value=True)
include_types = st.checkbox("Data Types (PNG)", value=True)
include_missing = st.checkbox("Missing Values Chart(PNG)", value=True)
include_ml_pred = st.checkbox("ML Prediction (Excel/CSV)", value=True)
include_ml_eval = st.checkbox("Model Evaluation plot (PNG)", value=True)

if st.button("Download ZIP"):
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            # cleaned dataset
            if include_clean:
                cleaned_df = df.dropna().drop_duplicates()
                zf.writestr("cleaned_dataset.csv", cleaned_df.to_csv(index=False))

            # correlation
            if include_corr:
                zf.writestr("correlation_matrix.csv", df.corr(numeric_only=True).to_csv())

            # saved visualization (png)
            if include_plots and st.session_state["plot_buf"] is not None:
                zf.writestr("visualization.png", st.session_state["plot_buf"].getvalue())

            # data types png
            if include_types:
                fig_t, ax_t = plt.subplots(figsize=(6, 4))
                df.dtypes.value_counts().plot(kind="bar", ax=ax_t)
                ax_t.set_title("Data Types Count")
                buf_t = fig_to_bytes(fig_t)
                zf.writestr("data_types.png", buf_t.getvalue())

            # missing values png
            if include_missing:
                fig_m, ax_m = plt.subplots(figsize=(6, 4))
                sns.heatmap(df.isnull(), cbar=False, ax=ax_m)
                buf_m = fig_to_bytes(fig_m)
                zf.writestr("missing_values.png", buf_m.getvalue())

            # ML prediction file
            if include_ml_pred and st.session_state.get("pred_buf") is not None:
                # name by preference: try xlsx if it was saved as Excel (detect by first bytes? we simply provide .xlsx and .csv)
                # We'll include both - if user wants Excel they can use the .xlsx, else .csv exists too
                try:
                    # if buffer contains XLSX (opened with ExcelWriter openpyxl), we'll write as xlsx
                    zf.writestr("ml_predictions.xlsx", st.session_state["pred_buf"].getvalue())
                except Exception:
                    # fallback to csv (text)
                    zf.writestr("ml_predictions.csv", st.session_state["pred_buf"].getvalue())

            # ML evaluation plot
            if include_ml_eval and st.session_state.get("ml_plot_buf") is not None:
                zf.writestr("model_evaluation.png", st.session_state["ml_plot_buf"].getvalue())

        zip_buffer.seek(0)
        st.download_button(
            label="📦 Download All Selected Files as ZIP",
            data=zip_buffer.getvalue(),
            file_name="all_data.zip",
            mime="application/zip",
        )
    except Exception as e:
        st.error("Error creating ZIP: " + str(e))
        st.text(traceback.format_exc())
