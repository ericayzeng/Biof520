import os
import pandas as pd
import numpy as np

def preprocess_uromol(df):
    """Preprocess UROMOL dataset.
    """
    # make names lowercase
    df.columns = df.columns.str.lower()
    
    # save ID column
    id_col = df["uromol.id"] if "uromol.id" in df.columns else pd.Series(np.nan, index=df.index)
        
    # drop if no recurrence information
    df = df.dropna(subset=['recurrence'])

    # filter to target patient population (low grade, TA)
    mask = (df['tumor.grade'].str.lower() == 'low grade') & (df['tumor.stage'].str.upper() == 'TA')
    df = df[mask].copy()

    # drop non-numeric columns
    keep = df.select_dtypes(include=['number']).columns.tolist()
    df = df[keep]
    
    # add id back in
    df.insert(0, "id", id_col.loc[df.index])

    return df

def preprocess_knowles(df):
    """Preprocess Knowles dataset.
    """
    # make names lowercase
    df.columns = df.columns.str.lower()
    
    # save ID column
    id_col = df["knowles_id"] if "knowles_id" in df.columns else pd.Series(np.nan, index=df.index)

    # Fill missing RFS time using futime_days. (converted from days to months)
    if "rfs_time" in df.columns and "futime_days." in df.columns:
        df["rfs_time"] = df["rfs_time"].fillna(df["futime_days."] / 30)
        
    # drop if no recurrence information
    df = df.dropna(subset=['recurrence'])

    # drop non-numeric columns
    keep = df.select_dtypes(include=['number']).columns.tolist()
    df = df[keep]
    
    # add id back in
    df.insert(0, "id", id_col.loc[df.index])

    return df

def main():
    # csv paths
    uromol_csv_path = os.path.join("data", "UROMOL_TaLG.csv")
    knowles_csv_path = os.path.join("data", "knowles_matched_TaLG_final.csv")

    # load data from csvs
    uromol_df = pd.read_csv(uromol_csv_path, low_memory=False)
    knowles_df = pd.read_csv(knowles_csv_path)

    # Preprocess
    uromol_clean = preprocess_uromol(uromol_df)
    knowles_clean = preprocess_knowles(knowles_df)
    
    # align datasets for generalizability
    common_columns = set(uromol_clean.columns) & set(knowles_clean.columns)
    common_columns = sorted(list(common_columns))
    uromol_clean = uromol_clean[common_columns]
    knowles_clean = knowles_clean[common_columns]

    # save as csv
    uromol_clean.to_csv("data/uromol_preprocessed.csv", index=False)
    knowles_clean.to_csv("data/knowles_preprocessed.csv", index=False)

if __name__ == "__main__":
    main()