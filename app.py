import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent

st.set_page_config(page_title="AI Powered Data Analyst Agent", layout="wide")

st.title("🤖 AI-Powered Data Analyst Agent")
st.write("Upload your dataset or use the default Superstore dataset to perform automated EDA, generate charts, and chat with your data!")

GOOGLE_API_KEY = st.sidebar.text_input("Enter Google API Key", type="password")
GROQ_API_KEY = st.sidebar.text_input("Enter Groq API Key", type="password")

if not GOOGLE_API_KEY or not GROQ_API_KEY:
    st.warning("Please provide both Google and Groq API keys in the sidebar to proceed.")
    st.stop()

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

groq_llm = ChatGroq(
    model="qwen-2.5-coder-32b-instruct",
    api_key=GROQ_API_KEY
)

def temp_tool():
    """This is just a dummy tool"""
    return "Hello world"

agent = create_agent(
    model=gemini_llm,
    tools=[temp_tool]
)

uploaded_file = st.sidebar.file_uploader("Upload CSV, XLS, or XLSX file", type=["csv", "xlsx", "xls"])
use_default = st.sidebar.checkbox("Use Default Superstore Dataset", value=True)

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success("Uploaded file loaded successfully!")
elif use_default:
    url = 'https://raw.githubusercontent.com/axisgras-hash/DATASETS/refs/heads/main/Superstore.csv'
    df = pd.read_csv(url)
    st.success("Default Superstore dataset loaded successfully!")
else:
    st.info("Please upload a dataset or check the default dataset option.")
    st.stop()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

if st.button("Run Comprehensive AI EDA"):
    with st.spinner("Agent is generating analysis code and performing EDA..."):
        try:
            df_sample = df.sample(min(5, len(df)))
            prompt = f"""You are a data analyst. Perform basic eda python single function perform_eda code and give all required analysis like missing values and columns. Data frame sample: {df_sample} data stats: {df_sample.describe()}"""
            
            response = agent.invoke({'messages': [{'role': 'user', 'content': prompt}]})
            ans = response["messages"][-1].content[-1]['text']
            code = ans.split("```")[1]
            if code.startswith("python"):
                code = code[6:]
            with open('basic_eda.py', 'w') as f:
                f.write(code)

            advance_prompt = """give detailed prompt for advance data analysis, which must include describe, corr, univariate numerical and object column analysis bivariate analysis, time series if any date column given multivariate analysis to perform different col like example sales, region, segment using bar plot with hue, give code with strict python and module code with pip install for any unknown new module if required"""

            response = agent.invoke({'messages': [{'role': 'user', 'content': advance_prompt}]})
            system_prompt_model = response["messages"][-1].content[-1]['text']

            new_prompt = """Give Python advance_eda.py file with every code inside a single function eda_by_ai and no need to load file, df is already loaded, starts with using df and any notes with comment""" + system_prompt_model

            response = agent.invoke({'messages': [{'role': 'user', 'content': new_prompt}]})
            ans = response["messages"][-1].content[-1]['text']
            code = ans.split("```")[1]
            if code.startswith("python"):
                code = code[6:]
            with open('advance_eda.py', 'w') as f:
                f.write(code)

            st.success("EDA scripts generated successfully!")

            from basic_eda import perform_eda
            from advance_eda import eda_by_ai

            st.subheader("📋 Basic EDA Report")
            try:
                basic_result = perform_eda(df)
                if isinstance(basic_result, str):
                    st.write(basic_result)
                else:
                    st.write(basic_result)
            except Exception as e:
                st.write("Executed basic EDA analysis directly on dataset:")
                st.write(df.describe())
                st.write("Missing Values:", df.isnull().sum())

            st.subheader("📈 Advanced EDA & Visualizations")
            try:
                eda_by_ai(df)
                for fig_num in plt.get_fignums():
                    st.pyplot(plt.figure(fig_num))
            except Exception as e:
                st.write("Fallback Advanced Visualizations:")
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(numeric_cols) > 0:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(df[numeric_cols[0]], kde=True, ax=ax)
                    st.pyplot(fig)
                
                if len(numeric_cols) >= 2:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during AI EDA execution: {e}")

st.subheader("💬 Chat with your Data")
user_query = st.text_input("Ask any question about your dataset:")

if user_query:
    with st.spinner("AI Agent is analyzing your query..."):
        chat_prompt = f"""You are an expert data analyst assistant. Answer the user question based on the dataframe context.
        Dataset columns: {list(df.columns)}
        Dataset head: {df.head(3).to_string()}
        User Query: {user_query}
        Provide a clear, direct, and insightful answer."""
        
        response = agent.invoke({'messages': [{'role': 'user', 'content': chat_prompt}]})
        chat_answer = response["messages"][-1].content[-1]['text']
        st.write("### Answer:")
        st.write(chat_answer)
