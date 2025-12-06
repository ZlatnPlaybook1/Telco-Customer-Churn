import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads CSV data into pandas DataFrame.

    Args: 
        file_path (str): Path to CSV file.
    
    Returns:
        pd.DataFrame loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.DataFrame(file_path)
