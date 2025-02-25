import os
import sys  # For exceptions
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # used for creating class variable

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts', 'train.csv')
    test_data_path : str = os.path.join('artifacts', 'test.csv')
    raw_data_path : str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        '''
        Used to setup the config variables. So that there is a defined path for saving the data.
        '''
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_ingestion(self):
        logging.info("Entered the data ingestion method")

        try :
            # here only we can read from any source
            df = pd.read_csv('Notebook/Data/stud_data.csv')
            logging.info("Read the dataset as DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train Test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state= len(df))

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of the data completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e: 
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_ingestion()    