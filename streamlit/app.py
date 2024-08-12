import streamlit as st
import pandas as pd
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull, save_model as reg_save_model
from pycaret.classification import setup as clf_setup, compare_models as clf_compare_models, pull as clf_pull, save_model as clf_save_model
from pycaret.clustering import setup as clu_setup, compare_models as clu_compare_models, pull as clu_pull, save_model as clu_save_model
from pycaret.time_series import setup as ts_setup, compare_models as ts_compare_models, pull as ts_pull, save_model as ts_save_model

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8637/8637099.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit")

st.write("Hello, world!")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload your Data for Modeling!")
    file = st.file_uploader('Upload your dataset here')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
    
if choice == "Profiling":
    st.title("Automated EDA")
    profile_report = ydata_profiling.ProfileReport(df)
    st_profile_report(profile_report)

if choice == "ML":
    st.title("Machine Learning")
    task_type = st.selectbox('Select the ML Task', ['Regression', 'Classification', 'Clustering', 'Time Series'])
    target = st.selectbox('Select your target', df.columns)
    
    if st.button("Train Model"):
        if task_type == 'Regression':
            reg_setup(df, target=target, silent=True)
            setup_df = reg_pull()
            st.info("This is the ML experiment settings")
            st.dataframe(setup_df)
            best_model = reg_compare_models()
            compare_df = reg_pull()
            st.info('This is the ML Model')
            st.dataframe(compare_df)
            reg_save_model(best_model, "best_model")

        elif task_type == 'Classification':
            clf_setup(df, target=target, silent=True)
            setup_df = clf_pull()
            st.info("This is the ML experiment settings")
            st.dataframe(setup_df)
            best_model = clf_compare_models()
            compare_df = clf_pull()
            st.info('This is the ML Model')
            st.dataframe(compare_df)
            clf_save_model(best_model, "best_model")

        elif task_type == 'Clustering':
            clu_setup(df, silent=True)
            setup_df = clu_pull()
            st.info("This is the ML experiment settings")
            st.dataframe(setup_df)
            best_model = clu_compare_models()
            compare_df = clu_pull()
            st.info('This is the ML Model')
            st.dataframe(compare_df)
            clu_save_model(best_model, "best_model")

        elif task_type == 'Time Series':
            ts_setup(df, target=target, silent=True)
            setup_df = ts_pull()
            st.info("This is the ML experiment settings")
            st.dataframe(setup_df)
            best_model = ts_compare_models()
            compare_df = ts_pull()
            st.info('This is the ML Model')
            st.dataframe(compare_df)
            ts_save_model(best_model, "best_model")

if choice == "Download":
    with open('best_model.pkl', "rb") as f:
        st.download_button("Download the Model", f, "best_model.pkl")
