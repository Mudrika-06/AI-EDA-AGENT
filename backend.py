# Step 3: Load all modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import langchain
import langchain_community
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
import streamlit as st
print("Modules Loaded Successfully!!")

# Step 4: Model creation
GOOGLE_API_KEY = "API KEY"
GROQ_API_KEY = "GROQ API KEY"

gemini_llm = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite",
    google_api_key = GOOGLE_API_KEY
    )

groq_llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key = GROQ_API_KEY
    )

# step 5: Agent Creation
def temp_tool():
  """This is just a dummy tool"""
  return "Hello world"

from langchain.agents import create_agent
agent = create_agent(
    model = gemini_llm,
    tools = [temp_tool] # we will give multiple tools here
    )


def load_dataset(path:str, agent = agent):
  """This Function help to take uploaded
  file path or url, and call agent
  here agent is responsible to return python code
  that helps to read uploaded dataset
  like CSV, XLSX, XLS, from Github url that ends
  with csv.
  System instructons: Don't give any reboot or harm code
  give correct executable code only
  example:
  df = pd.read_csv(file_path) or uploaded bytes string
  when file uploaded by streamlit or direct"""

  prompt = """return python code to read file
              in pandas using uploaded file extension"""


  try:
    import os
    if 'file_loader.py' not in os.getcwd():
      response = agent.invoke({'messages':[{'role':'user','content':prompt}]})
      ans = response["messages"][-1].content[-1]['text']
      code = ans.split("```")[1]
      with open('file_loader.py', 'w') as f:
        f.write(code[6:])

  except Exception as e:
    print(e)

    # with open('file_loader.py', 'w') as f:
    #   f.write(code[6:])


  return "Success"

  # # time.sleep(3)
  #from file_loader import read_uploaded_file
 # return read_uploaded_file(path)


url = 'https://raw.githubusercontent.com/axisgras-hash/DATASETS/refs/heads/main/Superstore.csv'
ans = load_dataset(url, agent)
ans



def read_file(path):
  """Read file using file_loader.py module
  read_uploaded_file function"""

  from file_loader import read_uploaded_file
  return read_uploaded_file(path)




# tool 3
def perform_eda_func(data, agent):
  """This Function takes data as an input
  and return basic eda for this given dataset
  pick data sample and perform analysis
  like shape, size etc, give code for that,
  agent will execute and get answer
  """

  df = data.sample(5)
  prompt = f"""You are a data analysts perform
  basic eda python single function perform_eda
  code and give all required
  analysis like missing values and columns
  Data Frame sample: {df}
  data stats: {df.describe()}"""

  response = agent.invoke({'messages':[{'role':'user','content':prompt}]})
  ans = response["messages"][-1].content[-1]['text']
  code = ans.split("```")[1]

  with open('basic_eda.py', 'w') as f:
    f.write(code[6:])


  # ============Advance EDA========

  advance_prompt = """give detailed prompt for
  advance data analysis, which must include
  describe, corr, univariate numerical and object column analysis
  bivariate analysis, time series if any date column given
  multivariate analysis to perform
  different col like example sales, region, segment
  using bar plot with hue, give code with strict python
  and module code with pip install for any unknown new module
  if required"""

  response = agent.invoke({'messages':[{'role':'user','content':advance_prompt}]})
  system_prompt_model = response["messages"][-1].content[-1]['text']

  # ========================The above will give detailed prompt only===========

  new_prompt = """Give Python advance_eda.py file with every code inside function eda_by_ai
  and no need to load file, df is already loaded, starts with using df
  and any notes with comment""" + system_prompt_model

  response = agent.invoke({'messages':[{'role':'user','content':new_prompt}]})
  ans = response["messages"][-1].content[-1]['text']
  code = ans.split("```")[1]


  with open('advance_eda.py', 'w') as f:
    f.write(code[6:])

  return "Success"


df = read_file(url)
df

from basic_eda import perform_eda
from advance_eda import eda_by_ai
