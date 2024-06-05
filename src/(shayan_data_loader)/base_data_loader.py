import logging
import pandas as pd

from common import constants, db_config

from .helper import (
    convert_date,
    merge_text,
    merge_interactions,
)


logger = logging.getLogger(constants.LOGGER_NAME)


class BaseDataLoader:
    """Base class for data loaders."""

    def __init__(self, file_path: str, topic: str, platform: str):
        self.file_path = file_path
        self.topic = topic
        self.platform = platform

    def load_data(self):
        """Load data from a given file path."""
        column_mapping = constants.COLUMN_CONFIG[self.platform]
        try:
            self.data = pd.read_csv(
                self.file_path, usecols=column_mapping.keys(), low_memory=False
            )
            self.data = self.data.rename(columns=column_mapping)
            logger.info(
                f"Successfully loaded {self.platform} data from {self.file_path}"
            )
        except Exception as e:
            logger.error(f"Error loading {self.platform} data: {e}")

    def transform_data(self):
        """Transform the raw data."""

        # drop nan values for id, author_id, and date
        self.data = self.data.dropna(subset=["id", "author_id", "date"]).reset_index(
            drop=True
        )
        logger.info(f"Data shape after dropping nan values: {self.data.shape}")

        # drop duplicates
        self.data = self.data.drop_duplicates(subset=["id"]).reset_index(drop=True)

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
        logger.info(f"Data shape after selecting time range: {self.data.shape}")
        logger.info(f"Min date: {self.data['date'].min()}")
        logger.info(f"Max date: {self.data['date'].max()}")
        logger.info("Processed date column")

        # Process text columns
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

    def get_chunks(self, chunksize):
        """Yield successive n-sized chunks from the dataframe."""
        for i in range(0, len(self.data), chunksize):
            yield self.data[i : i + chunksize]

    def save_to_db(self, chunksize=1000):
        """Save the data to the database."""

        if self.platform == "news":
            columns = constants.UNIFIED_COLUMNS["news"]
        else:
            columns = constants.UNIFIED_COLUMNS[self.topic]
        db = db_config.Database()
        db.create_table(
            schema_name=self.topic,
            table_name=self.platform,
            columns=columns,
        )

        logger.info(
            f"Inserting {len(self.data)} rows into {self.topic}.{self.platform} table"
        )

        chunks = self.get_chunks(chunksize)
        for chunk_index, chunk in enumerate(chunks):
            try:
                db.insert_data(
                    schema_name=self.topic, table_name=self.platform, chunk=chunk
                )
            except Exception as e:
                first_row_id = chunk.iloc[0]["id"]
                logger.error(
                    f"Error inserting batch {chunk_index + 1} starting with row id {first_row_id}: {e}"
                )
        db.close()

        logger.info(f"Data insertion completed")

    def process_data(self):
        # load data
        self.load_data()

        # transform data
        self.transform_data()

        # save data to database
        self.save_to_db()
