import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException

import os

print("FILE PATH:", __file__)

class proj1Data:
    """
    A class to export MongoDB records as a pandas DataFrame.
    """

    def __init__(self) -> None:
        """
        Initializes the MongoDB client connection.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str, database_name=None):

        if database_name is None:
            collection = self.mongo_client.database[collection_name]
        else:
            collection = self.mongo_client.client[database_name][collection_name]
        
        records = list(collection.find())
        print("Records:", len(records))

        df = pd.DataFrame(records)
        print("Shape:", df.shape)
        print("=" * 60)
        print("Database:", self.mongo_client.database.name)
        print("Collection:", collection.name)
        print("Count:", collection.count_documents({}))       
        return df