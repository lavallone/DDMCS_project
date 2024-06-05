import os
import pandas as pd
import logging
from common import constants
from .base_data_loader import BaseDataLoader
from .helper import (
    convert_date,
    merge_text,
    merge_interactions,
)

logger = logging.getLogger(constants.LOGGER_NAME)


class TwitterDataLoader(BaseDataLoader):
    def __init__(self, file_path: str, topic: str, platform: str):
        super().__init__(file_path, topic, platform)
        self.file_names = [f for f in os.listdir(self.file_path) if f.endswith(".csv")]
        logger.info(f"Found {len(self.file_names)} files in {self.file_path}")

    def load_data(self):
        dataframes = []
        # total_files = len(self.file_names)
        for i, file in enumerate(self.file_names):
            # logger.info(f"Loading file {i+1}/{total_files}")
            full_path = f"{self.file_path}/{file}"
            try:
                df = self.load_single_tw_file(full_path)
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Error loading {self.platform} data: {e}")
                continue
        self.data = pd.concat(dataframes, ignore_index=True)
        self.data = self.data.drop_duplicates(subset=["id"]).reset_index(drop=True)
        logger.info(f"Total records: {len(self.data)}")
        logger.info(f"Successfully loaded {self.platform} data from {self.file_path}")

    def load_single_tw_file(self, file_path: str):
        column_mapping = constants.COLUMN_CONFIG[self.platform]
        dtypes = {"id": str, "author_id": str}
        df = pd.read_csv(
            file_path, usecols=column_mapping.keys(), low_memory=False, dtype=dtypes
        )
        # to rename the name of the columns
        df.rename(columns=column_mapping, inplace=True)
        return df

    def transform_data(self):
        """Transform the raw data."""

        logger.info(f"Twitter data shape: {self.data.shape}")
        # drop nan values for id, author_id, and date
        self.data = self.data.dropna(subset=["id", "author_id", "date"]).reset_index(
            drop=True
        )
        logger.info(f"Twitter data shape after dropping nan: {self.data.shape}")

        # drop retweets
        # if the retweeted_id column is not null, then it's a retweet
        self.data = self.data[self.data["retweeted_id"].isnull()]
        logger.info(f"Twitter data shape after dropping retweets: {self.data.shape}")
        self.data = self.data.drop(columns=["retweeted_id"])

        # drop invalid ids
        invalid_id_mask = ~self.data["id"].str.isdigit()
        invalid_author_id_mask = ~self.data["author_id"].str.isdigit()
        invalid_masks = invalid_id_mask | invalid_author_id_mask
        invalid_rows = self.data[invalid_masks]
        for _, row in invalid_rows.iterrows():
            logger.warning(f"INVALID RECORD: {row.to_dict()}")
        self.data = self.data[~invalid_masks]
        logger.info(f"Twitter data shape after dropping invalid ids: {self.data.shape}")

        # double check the integrity of the data
        self.data = self.check_twitter_data_integrity(self.data)
        logger.info(f"Twitter data shape after checking integrity: {self.data.shape}")

        # convert date column
        self.data["date"] = self.data["date"].apply(convert_date)
        self.data = self.data.dropna(subset=["date"])
        self.data = self.data.sort_values(by=["date"])

        # select time range
        start_date = convert_date(constants.DATE_RANGE[self.topic]["start"])
        end_date = convert_date(constants.DATE_RANGE[self.topic]["end"])
        self.data = self.data[
            (self.data["date"] >= start_date) & (self.data["date"] <= end_date)
        ]
        logger.info(f"Twitter data shape after selecting time range: {self.data.shape}")
        logger.info(f"Min date: {self.data['date'].min()}")
        logger.info(f"Max date: {self.data['date'].max()}")
        logger.info("Processed date column")

        if self.topic == "chatgpt":
            # Merge text columns
            self.data = merge_text(self.data)
            logger.info("Processed text columns")
        else:
            # drop all the text columns for covid data
            text_columns = [col for col in self.data.columns if "text" in col]
            self.data = self.data.drop(columns=text_columns)

        # Merge interaction columns
        self.data = merge_interactions(self.data)
        logger.info("Processed interaction columns")

    @staticmethod
    def check_twitter_data_integrity(data):
        """
        Check the integrity of the data in 'id' and 'author_id' columns. Log and drop rows that don't meet criteria.
        """
        invalid_rows = []
        for column in ["id", "author_id"]:
            for index, value in data[column].items():
                if not isinstance(value, str):
                    logger.warning(
                        f"Invalid data in column '{column}' at index {index}: {value} is not a string."
                    )
                    invalid_rows.append(index)
                elif "e" in value:
                    logger.warning(
                        f"Invalid data in column '{column}' at index {index}: {value} has scientific notation."
                    )
                    invalid_rows.append(index)
                elif "." in value:
                    logger.warning(
                        f"Invalid data in column '{column}' at index {index}: {value} has a dot."
                    )
                    invalid_rows.append(index)
        logger.info(f"Found {len(invalid_rows)} invalid rows")
        return data.drop(invalid_rows).reset_index(drop=True)
