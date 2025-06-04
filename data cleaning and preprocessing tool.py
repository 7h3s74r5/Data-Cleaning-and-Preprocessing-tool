import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def load_data(file_path):
   if file_path.endswith('.csv'):
      return pd.read_csv(file_path)
   elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
   else:
        raise ValueError("Unsupported file format")


def handle_missing_data(df, strategy='mean', columns=None):
    cols = columns if columns else df.columns

    if strategy == 'drop':
        df.dropna(subset=cols, inplace=True)
    elif strategy in ['mean', 'median', 'mode']:
        for col in cols:
            if df[col].isnull().sum() == 0:
                continue
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:  # mode for all data types
                df[col].fillna(df[col].mode()[0], inplace=True)
    elif strategy in ['ffill', 'bfill']:
        df.fillna(method=strategy, inplace=True)
    else:
        raise ValueError("Invalid strategy. Use 'mean', 'median', 'mode', 'ffill', 'bfill', or 'drop'.")

    return df


def remove_duplicates(df):
    return df.drop_duplicates()

def handle_outliers(df, method='iqr', strategy='remove'):
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if strategy == 'remove':
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        elif strategy == 'cap':
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        else:
            raise ValueError("Invalid strategy. Use 'remove' or 'cap'.")

    return df


def save_data(df, path='cleaned_data.csv'):
    df.to_csv(path)

def clean_data_pipeline(filepath):
    df = pd.read_csv(filepath)
    df = remove_duplicates(df)
    df = handle_missing_data(df)
    df = handle_outliers(df)
    save_data(df)
    print("Data cleaning complete. Saved to 'cleaned_data.csv'")
    
    

filepath= input ("Enter the file path- ")
clean_data_pipeline(filepath)
