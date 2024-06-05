from common import logging_config, constants

logger = logging_config.setup_logger(constants.LOGGER_NAME)

import logging
import pandas as pd
from common import constants, db_config
from psycopg2 import sql

logger = logging.getLogger(constants.LOGGER_NAME)


class DailyUsers:
    """
    Load data for each topic-platform from the database and counts the number of unique users up to each day.
    All the results are saved in the processed.daily_user_counts table.

    To use this class, run scripts/daily_users_script.py
    """

    def __init__(self, schema_name: str, table_name: str):
        self.schema_name = schema_name
        self.table_name = table_name
        self.db = db_config.Database()
        # if table does not exist, create it
        self.db.create_table(
            "processed",
            "daily_user_counts",
            constants.UNIFIED_COLUMNS["daily_user_counts"],
        )

    def process_daily_user_counts(self):
        if self.table_name == "news":
            query = sql.SQL(
                """
                SELECT date, id AS author_id
                FROM {}.{}
                ORDER BY date
                """
            ).format(sql.Identifier(self.schema_name), sql.Identifier(self.table_name))
        else:
            query = sql.SQL(
                """
                SELECT date, author_id
                FROM {}.{}
                WHERE author_id IS NOT NULL
                ORDER BY date
                """
            ).format(sql.Identifier(self.schema_name), sql.Identifier(self.table_name))

        # Load the result into a DataFrame
        self.db.cursor.execute(query)
        result = self.db.cursor.fetchall()
        data = pd.DataFrame(result, columns=["date", "author_id"])
        logger.info(
            f"Loaded {len(data)} rows from {self.schema_name}.{self.table_name}"
        )

        # Count unique users up to each date
        data.sort_values(by=["date"], inplace=True)
        seen_users = set()
        results = []

        for date, group in data.groupby("date"):
            seen_users.update(group["author_id"].tolist())
            results.append(
                {
                    "date": date,
                    "cumulative_unique_users": len(seen_users),
                    "platform": self.table_name,
                    "topic": self.schema_name,
                }
            )

        results = pd.DataFrame(results)

        # Insert the results into the database
        insert_query = """
            INSERT INTO processed.daily_user_counts (date, cumulative_unique_users, platform, topic)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (date, platform, topic) DO NOTHING;
        """
        self.db.cursor.executemany(
            insert_query,
            results[["date", "cumulative_unique_users", "platform", "topic"]].values,
        )
        self.db.connection.commit()
        logger.info(
            f"Processed daily user counts for {self.table_name} and saved to processed.daily_user_counts"
        )
        self.db.close()


def count_daily_users():
    for topic in constants.TOPIC_TABLES:
        if topic["schema_name"] != "covid":
            continue
        logger.info(f"Processing {topic['schema_name']} schema")
        for table in topic["table_name"]:
            logger.info(f"Processing {topic['schema_name']}.{table} table")
            daily_posts = DailyUsers(topic["schema_name"], table)
            daily_posts.process_daily_user_counts()
            logger.info("-" * 100)
        logger.info("=" * 100)


if __name__ == "__main__":
    count_daily_users()
