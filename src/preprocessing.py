import numpy as np
import pandas as pd

RAW_DATA_PATH = "data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
OUTPUT_PATH = "data/CICDS_cleaned.csv"

METADATA_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]

LABEL_COL = "Label"


def load_raw(file_path=RAW_DATA_PATH):
    df = pd.read_csv(file_path, encoding="cp1252", low_memory=False)
    df.columns = df.columns.str.strip()
    df[LABEL_COL] = df[LABEL_COL].str.replace("\x96", "-").str.strip()
    return df


def clean(df):
    df = df.dropna(how="all").reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def drop_metadata(df, metadata_cols=METADATA_COLS):
    return df.drop(columns=metadata_cols)


def encode_labels(df):
    df["Label_binary"] = np.where(df[LABEL_COL] == "BENIGN", "BENIGN", "ATTACK")
    df["Label_encoded"] = (df["Label_binary"] == "ATTACK").astype(int)
    return df


def coerce_numeric(df):
    for col in df.columns:
        if col != LABEL_COL:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def handle_invalid_values(df):
    numeric_cols = df.columns[df.columns != LABEL_COL]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].where(df[numeric_cols] >= 0)
    return df


def impute_median(df):
    numeric_cols = df.columns[df.columns != LABEL_COL]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df


def preprocess(file_path=RAW_DATA_PATH, output_path=OUTPUT_PATH):
    df = load_raw(file_path)
    df = clean(df)
    n_before = df.shape[0]

    df = drop_metadata(df)
    df = coerce_numeric(df)
    df = handle_invalid_values(df)
    df = impute_median(df)
    df = encode_labels(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    string_label_cols = [LABEL_COL, "Label_binary"]
    feature_cols = [c for c in df.columns if c not in string_label_cols]
    assert not df[feature_cols].isna().any().any(), "NaNs remain in features"
    assert not np.isinf(df[numeric_cols]).any().any(), "Infs remain in features"

    df.to_csv(output_path, index=False)
    return df, n_before


if __name__ == "__main__":
    df, n_before = preprocess()

    print("Raw rows loaded (after blank-row + dup removal):", n_before)
    print("Dropped metadata columns:", METADATA_COLS)
    print("Final shape:", df.shape)
    print("Remaining NaNs:", int(df.isna().sum().sum()))
    print("Remaining infs:", int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum()))
    print()
    print("Label distribution (multiclass):")
    print(df[LABEL_COL].value_counts().to_string())
    print()
    print("Label distribution (binary):")
    print(df["Label_binary"].value_counts().to_string())
    print()
    print("Saved to:", OUTPUT_PATH)