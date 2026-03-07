import pandas as pd
import re
import os


def clean_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )
    return df

# df = pd.read_csv("raw_data.csv")
# df = clean_columns(df)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def rename_reports(file_path):
    for i, filename in enumerate(os.listdir(file_path)):
        new_name = f"report_{i}.pdf"
        os.rename(f"{file_path}/{filename}", f"{file_path}/{new_name}")
        