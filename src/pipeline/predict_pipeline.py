import sys
import os
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try :
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path= os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(model_path)
            prerpocessor = load_object(preprocessor_path)

            scaled_data = prerpocessor.transform(features)
            preds = model.predict(scaled_data)

            logging.info("Returning the predecitons")
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    '''
    This class will map all the features from the input to the values which we used in backend
    '''

    def __init__(self,
                 gender : str,
                 race_ethnicity : int,
                 parental_level_of_education : str,
                 lunch : int, 
                 test_preparation_course : int,
                 reading_score : int,
                 writing_score : int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_df(self):
        try : 
            custom_inp_data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
        
        except Exception as e:
            raise CustomException(e, sys)

        return pd.DataFrame(custom_inp_data)