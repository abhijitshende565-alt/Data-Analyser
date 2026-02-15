import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import io
import traceback
import zipfile
from newfeatures import search_by_word

<head>
<title>https://data-analyser-app.streamlit.app/.com homepage</title>
<meta name="<meta name="google-site-verification" content="GJsyKzaV37eA0sbSTna4pf11FuSzJliupwH7UfAIx4c" />
</head>

st.set_page_config(page_title="Data Analyser", layout="wide")

# ---------------- SESSION ----------------
if "df" not in st.session_state: st.session_state.df = None
if "plot_buf" not in st.session_state: st.session_state.plot_buf = None
if "pred_df" not in st.session_state: st.session_state.pred_df = None
if "pred_buf" not in st.session_state: st.session_state.pred_buf = None
if "ml_plot_buf" not in st.session_state: st.session_state.ml_plot_buf = None
if "chart_settings" not in st.session_state: st.session_state.chart_settings = {}
if "generate_chart" not in st.session_state: st.session_state.generate_chart = False
if "ml_done" not in st.session_state: st.session_state.ml_done = False
if "r2_value" not in st.session_state:
    st.session_state.r2_value = None

def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


st.title("📊 Data Analyser")

if st.button("Use Demo Data"):
    demo_data = {
        "Sales": [120, 150, 170, 200, 250],
        "Profit": [30, 50, 60, 90, 120],
        "Month": ["Jan", "Feb", "Mar", "Apr", "May"]
    }
    st.session_state.df = pd.DataFrame(demo_data)

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload Dataset", type=["csv","xlsx"])
if file is not None:
    if file.name.endswith(".csv",):
        st.session_state.df = pd.read_csv(file)
    else:
        st.session_state.df = pd.read_excel(file)
else:
    st.info("Upload file or use demo data")

df = st.session_state.df
if df is None: st.stop()

st.info(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# ---------------- DATA VIEW ----------------
st.subheader("View Data")
cols = st.multiselect("Columns", df.columns, default=df.columns)
rows = st.number_input("Rows to show",1,len(df),5)
st.dataframe(df[cols].head(rows), use_container_width=True)

search_by_word(df)

# ---------------- EDA ----------------
st.subheader("EDA")
st.dataframe(df.describe())

#---------------- Statistics-------------
st.subheader("📈 Statistics")
col1, col2 = st.columns(2)
with col1:
    st.write("*Missing Values:*")
    st.write(df.isnull().sum())
with col2:
    st.write("*Data Types:*")
    st.write(df.dtypes)

st.write("*Correlation Matrix (numeric):*")
st.dataframe(df.corr(numeric_only=True))

# ---------------- VISUALIZATION ----------------
st.subheader("Visualization")

theme = st.selectbox("Theme",["Light","Dark"])
graph_color = st.color_picker("Graph Color","#1f77b4")
bg_color = st.color_picker("Background","#ffffff")

x = st.selectbox("X",df.columns)
y = st.selectbox("Y",df.columns)
gtype = st.selectbox("Graph Type",["Line","Bar","Scatter","Histogram","Boxplot","Heatmap"])

st.session_state.chart_settings = {
    "theme": theme,
    "color": graph_color,
    "bg": bg_color,
    "x": x,
    "y": y,
    "type": gtype
}

if st.button("Generate Graph"):
    st.session_state.generate_chart = True

if st.session_state.generate_chart:
    s = st.session_state.chart_settings
    plt.style.use("dark_background" if s["theme"]=="Dark" else "default")

    fig, ax = plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor(s["bg"])
    ax.set_facecolor(s["bg"])

    if s["type"]=="Line":
        sns.lineplot(data=df,x=s["x"],y=s["y"],ax=ax,color=s["color"])
    elif s["type"]=="Bar":
        sns.barplot(data=df,x=s["x"],y=s["y"],ax=ax,color=s["color"])
    elif s["type"]=="Scatter":
        sns.scatterplot(data=df,x=s["x"],y=s["y"],ax=ax,color=s["color"])
    elif s["type"]=="Histogram":
        sns.histplot(df[s["x"]],ax=ax,color=s["color"])
    elif s["type"]=="Boxplot":
        sns.boxplot(data=df,x=s["x"],y=s["y"],ax=ax)
    elif s["type"]=="Heatmap":
        sns.heatmap(df.corr(numeric_only=True),annot=True,ax=ax)

    buf = fig_to_buf(fig)
    st.session_state.plot_buf = buf
    st.image(buf)

# ---------------- ML ----------------
st.subheader("Machine Learning")

nums = df.select_dtypes(include=np.number).columns
if len(nums)>=2:
    target = st.selectbox("Target",nums)
    features = st.multiselect("Features",[c for c in nums if c!=target])

    if st.button("Train & Predict"):
        X=df[features].dropna()
        y=df.loc[X.index,target]

        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2)
        model=LinearRegression().fit(Xtr,ytr)
        pred=model.predict(Xte)

        pdf=pd.DataFrame({"Actual":yte,"Predicted":pred})
        st.session_state.pred_df=pdf

        buf=io.BytesIO()
        pdf.to_excel(buf,index=False)
        buf.seek(0)
        st.session_state.pred_buf=buf

        fig,ax=plt.subplots()
        sns.scatterplot(x=yte,y=pred,ax=ax)
        st.session_state.ml_plot_buf=fig_to_buf(fig)

        r2 = r2_score(yte, pred)
        mae = mean_absolute_error(yte, pred)
        mse = mean_squared_error(yte, pred)

        st.session_state.r2_value = r2
        st.session_state.mae_value = mae
        st.session_state.mse_value = mse

        st.session_state.ml_done = True

# -------- SHOW ML ALWAYS --------
if st.session_state.ml_done:
    st.subheader("Prediction Result")
    st.dataframe(st.session_state.pred_df)
    st.image(st.session_state.ml_plot_buf)
    st.write("R2 Score:", st.session_state.r2_value)
    st.write("MAE:", st.session_state.mae_value)
    st.write("MSE:", st.session_state.mse_value)
    
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
                buf_t = fig_to_buf(fig_t)
                zf.writestr("data_types.png", buf_t.getvalue())

            # missing values png
            if include_missing:
                fig_m, ax_m = plt.subplots(figsize=(6, 4))
                sns.heatmap(df.isnull(), cbar=False, ax=ax_m)
                buf_m = fig_to_buf(fig_m)
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
