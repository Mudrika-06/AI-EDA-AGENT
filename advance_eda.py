
# ==========================================
# ADVANCED EDA PIPELINE (advance_eda.py)
# ==========================================

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')


def eda_by_ai(df):
    """Performs an exhaustive, advanced exploratory data analysis (EDA)

    on the provided DataFrame `df`.
    """
    # Set aesthetic style for plots
    sns.set_theme(style='whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)

    # ==========================================
    # 1. DATASET OVERVIEW & DESCRIPTION
    # ==========================================
    print('--- 1. DATASET OVERVIEW & DESCRIPTION ---')
    print(f'Dataset Shape: {df.shape}\n')
    print('Missing Values:\n', df.isnull().sum(), '\n')
    print('Numerical Summary:')
    print(df.describe())
    
    # Check if any object/categorical columns exist before describing
    object_cols_summary = df.select_dtypes(include=['O', 'category']).columns
    if len(object_cols_summary) > 0:
        print('\nCategorical Summary:')
        print(df.describe(include=['O', 'category']))
    else:
        print('\nCategorical Summary: No categorical (object/category) columns found.')

    # ==========================================
    # 2. CORRELATION ANALYSIS
    # ==========================================
    print('\n--- 2. CORRELATION ANALYSIS ---')
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        plt.figure(figsize=(8, 6))
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(
            corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5
        )
        plt.title('Correlation Matrix of Numerical Features')
        plt.show()

        # Highlight strong multicollinearity
        print('Checking for strong multicollinearity (|r| > 0.8):')
        high_corr_found = False
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_found = True
                    print(
                        f'  - High correlation between "{corr_matrix.columns[i]}" and "{corr_matrix.columns[j]}": {corr_matrix.iloc[i, j]:.2f}'
                    )
        if not high_corr_found:
            print('  - No strong multicollinearity detected (|r| > 0.8).')
    else:
        print('Not enough numerical columns to compute correlation matrix.')

    # ==========================================
    # 3. UNIVARIATE ANALYSIS
    # ==========================================
    print('\n--- 3. UNIVARIATE ANALYSIS ---')
    # Numerical Univariate
    if len(numerical_cols) > 0:
        print('Generating distribution and box plots for numerical columns...')
        for col in numerical_cols:
            # Print skewness and kurtosis
            print(
                f'  -> {col}: Skewness = {df[col].skew():.2f}, Kurtosis = {df[col].kurtosis():.2f}'
            )

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color='skyblue')
            axes[0].set_title(f'Distribution of {col}')
            sns.boxplot(x=df[col].dropna(), ax=axes[1], color='lightgreen')
            axes[1].set_title(f'Boxplot of {col}')
            plt.tight_layout()
            plt.show()
    else:
        print('No numerical columns found for univariate analysis.')

    # Object/Categorical Univariate
    object_cols = df.select_dtypes(include=['O', 'category']).columns
    if len(object_cols) > 0:
        print('Generating count plots for categorical columns...')
        for col in object_cols:
            plt.figure(figsize=(8, 4))
            order_counts = df[col].value_counts().iloc[
                :15
            ]  # Top 15 if high cardinality
            sns.countplot(
                data=df,
                x=col,
                order=order_counts.index,
                palette='viridis',
                hue=col,
                legend=False,
            )
            plt.title(f'Frequency Count of {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    else:
        print('No categorical columns found for univariate analysis.')

    # ==========================================
    # 4. BIVARIATE ANALYSIS & CATEGORICAL GROUPING
    # ==========================================
    print(
        '\n--- 4. BIVARIATE ANALYSIS (Bar plot with Hue: Sales by Region & Segment) ---'
    )
    if (
        'Sales' in df.columns
        and 'Region' in df.columns
        and 'Segment' in df.columns
    ):
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df,
            x='Region',
            y='Sales',
            hue='Segment',
            estimator=np.mean,
            palette='Set2',
            errorbar=None,
        )
        plt.title('Average Sales by Region and Segment')
        plt.ylabel('Average Sales')
        plt.xlabel('Region')
        plt.legend(title='Segment')
        plt.tight_layout()
        plt.show()
    else:
        print(
            "Columns 'Sales', 'Region', and/or 'Segment' not found. Skipping default bivariate example."
        )

    # ==========================================
    # 5. TIME SERIES ANALYSIS
    # ==========================================
    print('\n--- 5. TIME SERIES ANALYSIS ---')
    # Automatically inspect dataframe to identify date/time columns or objects that can be parsed
    date_cols = df.select_dtypes(
        include=['datetime64[ns]', 'datetimetz']
    ).columns

    if len(date_cols) == 0:
        # Try checking object columns that might represent dates
        for col in object_cols:
            try:
                pd.to_datetime(df[col], errors='raise')
                date_cols = [col]
                break
            except (ValueError, TypeError):
                continue

    if len(date_cols) > 0:
        date_col = date_cols[0]
        print(f'Detected Date Column: {date_col}')

        # Ensure datetime type
        temp_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(temp_df[date_col]):
            temp_df[date_col] = pd.to_datetime(
                temp_df[date_col], errors='coerce'
            )

        # Check if 'Sales' exists for time series aggregation, otherwise use the first numerical column
if 'Sales' in temp_df.columns:
    ts_metric = 'Sales'
else:
    ts_metric = (
        numerical_cols[0] if len(numerical_cols) > 0 else None
    )

        if ts_metric:
            # Aggregate by date (e.g., Monthly Sum)
            ts_df = (
                temp_df.dropna(subset=[date_col])
                .set_index(date_col)
                .resample('ME')[ts_metric]
                .sum()
                .reset_index(name=f'Total_{ts_metric}')
            )
            ts_df['Rolling_Mean_3M'] = (
                ts_df[f'Total_{ts_metric}'].rolling(window=3).mean()
            )

            plt.figure(figsize=(12, 5))
            plt.plot(
                ts_df[date_col],
                ts_df[f'Total_{ts_metric}'],
                label=f'Monthly {ts_metric}',
                marker='o',
                alpha=0.6,
            )
            plt.plot(
                ts_df[date_col],
                ts_df['Rolling_Mean_3M'],
                label='3-Month Rolling Mean',
                color='red',
                linewidth=2,
            )
            plt.title(
                f'Time Series Analysis: Monthly {ts_metric} Trend & Rolling Average'
            )
            plt.xlabel('Date')
            plt.ylabel(f'Total {ts_metric}')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print(
                'No suitable numerical metric available for time series aggregation.'
            )
    else:
        print('No Date column detected for Time Series Analysis.')

    # ==========================================
    # 6. MULTIVARIATE ANALYSIS
    # ==========================================
    print('\n--- 6. MULTIVARIATE ANALYSIS ---')
    if len(numerical_cols) >= 3 and len(object_cols) > 0:
        hue_col = object_cols[0]
        cols_to_plot = list(numerical_cols[:4]) + [hue_col]
        
        # Filter dataframe for valid columns and drop NAs for clean pairplot
        pairplot_df = df[cols_to_plot].dropna()
        
        sns.pairplot(pairplot_df, hue=hue_col, palette='husl', diag_kind='kde')
        plt.suptitle(
            f'Multivariate Pairplot colored by {hue_col}', y=1.02, fontsize=14
        )
        plt.show()
    else:
        print(
            'Skipping multivariate pairplot: requires at least 3 numerical columns and 1 categorical column.'
        )
