

import os
import pandas as pd


def read_uploaded_file(file_path):
    """Reads a file into a pandas DataFrame based on its file extension.

    Parameters:
    file_path (str): The path to the uploaded file.

    Returns:
    pd.DataFrame: The loaded data.
    """
    # Get the file extension and convert it to lowercase
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Dictionary mapping extensions to their respective pandas read functions
    readers = {
        ".csv": lambda f: pd.read_csv(f),
        ".txt": lambda f: pd.read_csv(
            f, sep=None, engine="python"
        ),  # Auto-detect separator for txt
        ".xls": lambda f: pd.read_excel(f),
        ".xlsx": lambda f: pd.read_excel(f),
        ".json": lambda f: pd.read_json(f),
        ".parquet": lambda f: pd.read_parquet(f),
        ".pkl": lambda f: pd.read_pickle(f),
        ".pickle": lambda f: pd.read_pickle(f),
    }

    if ext in readers:
        try:
            return readers[ext](file_path)
        except Exception as e:
            raise RuntimeError(f"Error reading {ext} file: {e}")
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported formats are: {list(readers.keys())}"
        )


# --- Example Usage ---
# df = read_uploaded_file("path/to/your/file.csv")
# print(df.head())
