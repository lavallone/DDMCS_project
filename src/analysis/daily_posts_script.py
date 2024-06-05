from common import logging_config, constants

logger = logging_config.setup_logger(constants.LOGGER_NAME)

import logging
from common import constants, db_config
from psycopg2 import sql

logger = logging.getLogger(constants.LOGGER_NAME)


class DailyPosts:
    """
    Load data for each topic-platform from the database and counts the (cumulative) number of posts per day.
    All the results are saved in the processed.daily_post_counts table.

    To use this class, run scripts/daily_posts_script.py
    """

    def __init__(self, schema_name: str, table_name: str):
        self.schema_name = schema_name
        self.table_name = table_name
        self.db = db_config.Database()
        # if table does not exist, create it
        self.db.create_table(
            "processed",
            "daily_post_counts",
            constants.UNIFIED_COLUMNS["daily_post_counts"],
        )

    def process_daily_post_counts(self):
        query = sql.SQL(
            """
            WITH daily_counts AS (
                SELECT
                    date,
                    COUNT(id) AS posts
                FROM
                    {}.{}
                GROUP BY
                    date
            )
            SELECT
                date,
                posts,
                SUM(posts) OVER (ORDER BY date) AS cumulative_posts
            FROM daily_counts
            ORDER BY
                date;
            """
        ).format(sql.Identifier(self.schema_name), sql.Identifier(self.table_name))

        self.db.cursor.execute(query)
        result = self.db.cursor.fetchall()

        # Add platform and topic values to the result
        formatted_results = [
            (date, post, cumulative, self.table_name, self.schema_name)
            for date, post, cumulative in result
        ]

        insert_query = """
            INSERT INTO processed.daily_post_counts (date, posts, cumulative_posts, platform, topic)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (date, platform, topic) DO NOTHING;        
            """

        self.db.cursor.executemany(insert_query, formatted_results)
        self.db.connection.commit()

        logger.info(
            f"Processed daily post counts for {self.table_name} and saved to processed.daily_post_counts"
        )
        self.db.close()

def count_daily_posts():
    for topic in constants.TOPIC_TABLES:
        logger.info(f"Processing {topic['schema_name']} schema")
        for table in topic["table_name"]:
            logger.info(f"Processing {topic['schema_name']}.{table} table")
            daily_posts = DailyPosts(topic["schema_name"], table)
            daily_posts.process_daily_post_counts()
            logger.info("-" * 100)
        logger.info("=" * 100)


if __name__ == "__main__":
    count_daily_posts()
