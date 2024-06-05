import re
import pandas as pd
from datetime import datetime


def convert_date(date_str: str) -> datetime.date:
    # Cleaning date string: removing timezone, we are interested in the date component.
    cleaned_str = re.sub(r"[A-Z]|T|Z", " ", date_str).strip()

    try:
        dt = pd.to_datetime(cleaned_str, errors="coerce")
        return dt.date()
    except:
        return None


def merge_text(data: pd.DataFrame) -> pd.DataFrame:
    text_columns = [col for col in data.columns if "text" in col.lower()]

    def process_text_cols(text_cols: pd.Series) -> str:
        text_values = [value for value in text_cols if pd.notna(value)]
        return " ".join(map(str, text_values))

    if len(text_columns) > 1:
        data["text"] = data[text_columns].apply(process_text_cols, axis=1)
        data = data.drop(columns=text_columns)
    else:
        data["text"] = data["text"].astype(str)

    return data


def merge_interactions(data: pd.DataFrame) -> pd.DataFrame:
    interaction_columns = [col for col in data.columns if "interaction" in col.lower()]

    def process_interaction_cols(interaction_cols: pd.Series) -> pd.Series:
        interaction_cols = interaction_cols.fillna(0)
        interaction_cols = interaction_cols.astype(str).str.replace(",", "")
        interaction_cols = interaction_cols.apply(
            lambda x: pd.to_numeric(x, errors="coerce")
        )
        return interaction_cols

    processed_interaction_cols = data[interaction_columns].apply(
        process_interaction_cols, axis=1
    )

    # If multiple interaction columns, sum them up
    if len(interaction_columns) > 1:
        data["interaction"] = processed_interaction_cols.sum(axis=1)
        data = data.drop(columns=interaction_columns)
    else:
        data["interaction"] = processed_interaction_cols[interaction_columns[0]]

    return data
