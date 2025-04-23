import os
import pandas as pd
import numpy as np

# Paths
RAW_DATA_DIR = "Resources/datasets"
PROCESSED_DATA_DIR = "Resources/processed_data"

def load_dataset(filename):
    path = os.path.join(RAW_DATA_DIR, filename)
    return pd.read_csv(path)

def clean_ckd_dataset_1(df):
    df = df.copy()

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(lambda x: str(x).strip().lower() if pd.notnull(x) else x)

    binary_mappings = {
        'yes': 1, 'no': 0,
        'present': 1, 'notpresent': 0,
        'poor': 0, 'good': 1,
        'ckd': 1, 'notckd': 0
    }
    df.replace(binary_mappings, inplace=True)
    df = df.infer_objects()

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass

    return df

def finalize_clean_ckd_dataset_1(df):
    df = df.copy()

    if 'rbc' in df.columns:
        df['rbc'] = df['rbc'].map({'normal': 1, 'abnormal': 0}).astype(float)
    if 'pc' in df.columns:
        df['pc'] = df['pc'].map({'normal': 1, 'abnormal': 0}).astype(float)

    fill_zero_cols = ['htn', 'dm', 'cad', 'ane', 'pe', 'pcc', 'ba']
    print("\nNulls before binary fill:")
    for col in fill_zero_cols + ['appet']:
        if col in df.columns:
            print(f"  {col}: {df[col].isnull().sum()} nulls")

    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    if 'appet' in df.columns:
        df['appet'] = df['appet'].fillna(1.0)

    print("\nNulls after binary fill:")
    for col in fill_zero_cols + ['appet']:
        if col in df.columns:
            print(f"  {col}: {df[col].isnull().sum()} nulls")

    for col in df.select_dtypes(include='number').columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df

def parse_numeric_range(value):
    try:
        if pd.isnull(value):
            return np.nan
        value = str(value).strip()
        if ' - ' in value:
            low, high = value.split(' - ')
            return (float(low) + float(high)) / 2
        if value.startswith('≥') or value.startswith('>'):
            return float(value[1:].strip())
        if value.startswith('<'):
            return float(value[1:].strip())
        return float(value)
    except:
        return np.nan

def clean_ckd_dataset_2():
    path = os.path.join(RAW_DATA_DIR, "ckd-dataset-v2.csv")
    raw = pd.read_csv(path, header=None, skiprows=3)
    raw.columns = pd.read_csv(path, nrows=1, header=None).iloc[0]

    for col in raw.columns:
        raw[col] = raw[col].apply(parse_numeric_range)

    raw.dropna(axis=1, thresh=len(raw) * 0.5, inplace=True)

    for col in raw.select_dtypes(include='number').columns:
        if raw[col].isnull().sum() > 0:
            raw[col] = raw[col].fillna(raw[col].median())

    return raw

def save_dataset(df, filename):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(path, index=False)

def main():
    print("Loading chronic_kidney_disease.csv...")
    df1 = load_dataset("chronic_kidney_disease.csv")

    print("Cleaning dataset 1...")
    df1_clean = clean_ckd_dataset_1(df1)
    df1_clean = finalize_clean_ckd_dataset_1(df1_clean)

    save_dataset(df1_clean, "processed_ckd_1.csv")

    print("Loading and cleaning ckd-dataset-v2.csv...")
    df2_clean = clean_ckd_dataset_2()
    save_dataset(df2_clean, "processed_ckd_2.csv")

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()

