
import numpy as np
import pandas as pd


def perform_eda(df: pd.DataFrame):
    """Performs basic Exploratory Data Analysis (EDA) on a given pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.
    """
    print("=" * 60)
    print(" 📊 EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 60)

    # 1. Dataset Shape
    print(f"\n[1] DATASET DIMENSIONS:")
    print(f"Number of Rows    : {df.shape[0]}")
    print(f"Number of Columns : {df.shape[1]}")

    # 2. Column Names & Data Types
    print(f"\n[2] COLUMN DATA TYPES & NON-NULL COUNTS:")
    print("-" * 50)
    df.info()

    # 3. Missing Values Summary
    print(f"\n[3] MISSING VALUES ANALYSIS:")
    print("-" * 50)
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().mean() * 100).round(2)

    missing_df = pd.DataFrame(
        {"Missing Values": missing_count, "Percentage (%)": missing_percent}
    )
    # Filter to show only columns with missing values, or show all if preferred
    print(missing_df[missing_df["Missing Values"] > 0])
    if missing_df["Missing Values"].sum() == 0:
        print("🎉 Great news! There are no missing values in this dataset.")

    # 4. Duplicate Rows
    print(f"\n[4] DUPLICATE ROWS:")
    print("-" * 50)
    duplicates = df.duplicated().sum()
    print(
        f"Number of duplicate rows: {duplicates} ({round((duplicates / len(df)) * 100, 2)}% of dataset)"
    )

    # 5. Statistical Summary (Numerical Columns)
    print(f"\n[5] STATISTICAL SUMMARY (Numerical Columns):")
    print("-" * 50)
    num_cols = df.select_dtypes(include=[np.number])
    if not num_cols.empty:
        display_summary = num_cols.describe().T[
            ["count", "mean", "std", "min", "50%", "max"]
        ]
        display_summary = display_summary.rename(
            columns={"50%": "median"}
        )  # Rename 50% to Median
        print(display_summary)
    else:
        print("No numerical columns found in the dataset.")

    # 6. Statistical Summary (Categorical Columns)
    print(f"\n[6] STATISTICAL SUMMARY (Categorical Columns):")
    print("-" * 50)
    cat_cols = df.select_dtypes(include=["object", "category"])
    if not cat_cols.empty:
        print(cat_cols.describe().T)
    else:
        print("No categorical columns found in the dataset.")

    print("\n" + "=" * 60)
    print(" END OF EDA REPORT")
    print("=" * 60)
