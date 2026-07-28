import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import io
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

st.set_page_config(page_title="AI Powered Data Analyst Agent", layout="wide")

st.title("🤖 AI-Powered Data Analyst Agent")
st.write("Upload your dataset or use the default URL to automatically perform EDA, generate univariate, bivariate, and multivariate charts, and chat with your data!")

with st.sidebar:
    st.header("🔑 API Configuration")
    GOOGLE_API_KEY = st.text_input("Google API Key", type="password")
    GROQ_API_KEY = st.text_input("Groq API Key", type="password")
    
    st.header("📁 Data Input")
    data_source = st.radio("Choose Data Source", ["Upload CSV/XLSX", "Use Default URL"])
    
    uploaded_file = None
    default_url = 'https://raw.githubusercontent.com/axisgras-hash/DATASETS/refs/heads/main/Superstore.csv'
    
    if data_source == "Upload CSV/XLSX":
        uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls"])
    else:
        st.write(f"Default URL: `{default_url}`")

if not GOOGLE_API_KEY:
    st.warning("Please enter your Google API Key in the sidebar to proceed.")
    st.stop()

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

groq_llm = ChatGroq(
    model="qwen-2.5-coder-32b-instruct",
    api_key=GROQ_API_KEY
) if GROQ_API_KEY else gemini_llm

def temp_tool():
    """This is just a dummy tool"""
    return "Hello world"

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI data analyst assistant. Provide only executable Python code block when asked for code."),
    ("human", "{input}"),
])

try:
    agent = create_tool_calling_agent(gemini_llm, [temp_tool], prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=[temp_tool], verbose=False)
except Exception:
    agent_executor = None

def load_dataset_dynamic(uploaded_file, url):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    else:
        try:
            df = pd.read_csv(url)
            return df
        except Exception as e:
            st.error(f"Error loading default URL: {e}")
            return None

df = load_dataset_dynamic(uploaded_file, default_url)

if df is not None:
    st.success("Dataset Loaded Successfully!")
    
    with st.expander("🔍 Preview Dataset"):
        st.dataframe(df.head(10))
        st.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Basic EDA", "📈 Advanced Visualizations", "📉 Auto EDA Execution", "💬 Chat with Data"])

    with tab1:
        st.subheader("Basic Exploratory Data Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        
        st.write("### Summary Statistics")
        st.dataframe(df.describe())
        
        st.write("### Missing Values per Column")
        missing_df = pd.DataFrame({'Missing Count': df.isnull().sum(), 'Percentage (%)': (df.isnull().sum() / len(df)) * 100})
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])

    with tab2:
        st.subheader("Univariate, Bivariate, and Multivariate Analysis Charts")
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        analysis_type = st.selectbox("Select Analysis Type", ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])
        
        if analysis_type == "Univariate Analysis":
            st.write("### Univariate Analysis")
            uni_col = st.selectbox("Select Column for Univariate Plot", df.columns)
            fig, ax = plt.subplots(figsize=(10, 5))
            if uni_col in numeric_cols:
                sns.histplot(df[uni_col].dropna(), kde=True, ax=ax, color='skyblue')
                ax.set_title(f'Distribution of {uni_col}')
            else:
                top_vals = df[uni_col].value_counts().head(10)
                sns.barplot(x=top_vals.index, y=top_vals.values, ax=ax, palette='viridis')
                ax.set_title(f'Top Categories in {uni_col}')
                plt.xticks(rotation=45)
            st.pyplot(fig)
            
        elif analysis_type == "Bivariate Analysis":
            st.write("### Bivariate Analysis")
            if len(numeric_cols) >= 2:
                col_x = st.selectbox("Select X Axis", df.columns, index=0)
                col_y = st.selectbox("Select Y Axis (Numeric)", numeric_cols, index=1)
                fig, ax = plt.subplots(figsize=(10, 5))
                if col_x in numeric_cols:
                    sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax, alpha=0.6)
                else:
                    sns.boxplot(data=df, x=col_x, y=col_y, ax=ax, palette='Set2')
                    plt.xticks(rotation=45)
                ax.set_title(f'{col_y} vs {col_x}')
                st.pyplot(fig)
            else:
                st.warning("Not enough numerical columns for bivariate analysis.")
                
        elif analysis_type == "Multivariate Analysis":
            st.write("### Multivariate Analysis (Correlation & Hue Plots)")
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                ax.set_title("Correlation Matrix Heatmap")
                st.pyplot(fig)
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                cat_hue = st.selectbox("Select Categorical Hue", categorical_cols)
                num_val = st.selectbox("Select Numeric Value", numeric_cols)
                fig, ax = plt.subplots(figsize=(10, 5))
                top_cat = df[categorical_cols[0]].value_counts().head(5).index
                filtered_df = df[df[categorical_cols[0]].isin(top_cat)]
                sns.barplot(data=filtered_df, x=categorical_cols[0], y=num_val, hue=cat_hue, ax=ax, ci=None)
                ax.set_title(f'Multivariate Bar Plot: {num_val} by {categorical_cols[0]} and {cat_hue}')
                plt.xticks(rotation=45)
                st.pyplot(fig)

    with tab3:
        st.subheader("Automated AI Code Execution for EDA")
        if st.button("Generate & Run AI EDA Script"):
            with st.spinner("AI is generating comprehensive EDA python code..."):
                prompt = f"""Write a standalone executable python script using pandas, matplotlib, seaborn to perform advanced EDA on dataframe `df`. Print out summaries, correlation matrix, and plot top distributions. Ensure all code is strictly enclosed in triple backticks."""
                
                try:
                    response = gemini_llm.invoke(prompt)
                    ans_text = response.content
                    if "```python" in ans_text:
                        code_block = ans_text.split("```python")[1].split("```")[0]
                    elif "```" in ans_text:
                        code_block = ans_text.split("```")[1].split("```")[0]
                    else:
                        code_block = ans_text
                        
                    st.code(code_block, language='python')
                    
                    st.write("### Execution Results:")
                    local_vars = {'df': df, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns}
                    exec(code_block, {}, local_vars)
                    st.success("AI EDA Code executed successfully!")
                except Exception as e:
                    st.error(f"Error executing AI generated code: {e}")

    with tab4:
        st.subheader("💬 Chat with your Dataset")
        st.write("Ask any question regarding insights, trends, or aggregations from your data.")
        
        user_query = st.text_input("Ask a question about the dataset:")
        if user_query:
            with st.spinner("Analyzing your query..."):
                data_sample = df.head(3).to_string()
                columns_info = str(df.dtypes.to_dict())
                chat_prompt = f"""You are an expert data analyst assistant. 
                The user is asking a question about a dataset. 
                Dataset columns and types: {columns_info}
                Dataset sample: {data_sample}
                User question: {user_query}
                Write executable pandas code snippet (assigning final answer to a variable named `result` or printing it) to answer the query, or provide a direct analytical text answer."""
                
                try:
                    chat_response = gemini_llm.invoke(chat_prompt)
                    st.write(chat_response.content)
                except Exception as e:
                    st.error(f"Error generating response: {e}")

else:
    st.info("Please upload a dataset or ensure the default URL is accessible.")