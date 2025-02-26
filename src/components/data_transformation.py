import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    '''
    Here we are defining the path where the objects will be stroed as .pkl format
    '''
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation : 
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.num_cols = ['reading_score', 'writing_score']
        self.cat_cols = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
    
    def get_data_transformer_object(self):
        '''
        This Function is responsible for transforming the data and return the Transformed columns

        Input : self -> default
        Output : Preprocessor : ColumnTransformer Class object.
        '''
        try:
            # Creating the pipeline
            num_pipeline = Pipeline(
                steps= [
                    ('Imputer', SimpleImputer(strategy='median')),
                    ("Scaler", StandardScaler())
                ]
            )
            logging.info("Numerical Columns Scaling Compelted")
            cat_pipline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy='most_frequent')),
                    ("One Hot Encoding", OneHotEncoder(drop='first')),
                    ("Scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical Columns Encoding Compelted")

            preprocessor = ColumnTransformer([
                ("Numerical Pipeline", num_pipeline, self.num_cols),
                ("Categorical Pipeline", cat_pipline, self.cat_cols)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading the train and test datasets completed")

            preprocess = self.get_data_transformer_object()
            logging.info("Obtained the Preprocessing object")

            # Creating target column
            target_col = 'math_score'
            
            input_train_df = train_df.drop(columns=[target_col])
            target_train = train_df[target_col]

            input_test_df = test_df.drop(columns=[target_col])
            target_test = test_df[target_col]

            input_train_arr = preprocess.fit_transform(input_train_df)
            input_test_arr = preprocess.transform(input_test_df)

            train_arr = np.c_[input_train_arr, np.array(target_train)]
            test_arr = np.c_[input_test_arr, np.array(target_test)]

            logging.info("Transformed the dataframes")

            # This is from the utils
            save_object(
                filepath = self.data_transformation_config.preprocessor_file_path,
                obj = preprocess
            )
            logging.info("Object saved Successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)