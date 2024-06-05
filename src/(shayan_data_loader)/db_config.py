import psycopg2
import json
import logging

from common import constants, directories
from psycopg2 import sql
from psycopg2.extras import execute_values
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logger = logging.getLogger(constants.LOGGER_NAME)

# load database config
with open(directories.DB_CONFIG, "r") as f:
    db_confg = json.load(f)


class Database:
    def __init__(self):
        self.connection = psycopg2.connect(**db_confg)
        logger.info("Database connection established")
        self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        self.cursor = self.connection.cursor()
        logger.info("Database cursor created")

    def create_table(self, schema_name, table_name, columns):
        # Check if table exists
        self.cursor.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = %s
            );
            """,
            (schema_name, table_name),
        )
        exists = self.cursor.fetchone()[0]

        if not exists:
            # If table does not exist, create it
            column_string = ",\n".join(columns)
            self.cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE {}.{} (
                    {}
                    )
                    """
                ).format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                    sql.SQL(column_string),
                )
            )
            logger.info(f"Table {schema_name}.{table_name} created.")
        else:
            logger.info(
                f"Table {schema_name}.{table_name} already exists. Did not create a new table."
            )

    def insert_data(self, schema_name, table_name, chunk):
        columns = ", ".join(chunk.columns)
        insert_query = sql.SQL("INSERT INTO {}.{} ({}) VALUES %s").format(
            sql.Identifier(schema_name), sql.Identifier(table_name), sql.SQL(columns)
        )
        values_list = [tuple(row) for index, row in chunk.iterrows()]
        execute_values(self.cursor, insert_query, values_list)
        self.connection.commit()
        logger.info(f"Inserted {len(chunk)} rows into {schema_name}.{table_name}")

    def update_data(self, schema_name, table_name, chunk, key_column):
        """
        Update data in the specified table based on the key_column.

        Args:
        - schema_name: Name of the schema
        - table_name: Name of the table
        - chunk: DataFrame containing data to be updated
        - key_column: Name of the primary key or unique column to determine which rows to update
        """

        # Generate the SET portion of the SQL statement
        columns = chunk.columns
        set_clause = ", ".join([f"{col} = %s" for col in columns if col != key_column])

        # Construct the UPDATE SQL statement
        update_query = sql.SQL("UPDATE {}.{} SET {} WHERE {} = %s").format(
            sql.Identifier(schema_name),
            sql.Identifier(table_name),
            sql.SQL(set_clause),
            sql.Identifier(key_column),
        )

        for _, row in chunk.iterrows():
            # Split the data into values to SET and the key value
            key_value = row[key_column]
            update_values = [row[col] for col in columns if col != key_column] + [
                key_value
            ]

            # Execute the query
            self.cursor.execute(update_query, update_values)

        self.connection.commit()
        logger.info(f"Updated {len(chunk)} rows in {schema_name}.{table_name}")

    def close(self):
        self.connection.commit()
        self.cursor.close()
        self.connection.close()
        logger.info("Database connection closed")
