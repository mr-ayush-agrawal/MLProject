import os
import sys
import dill
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(filepath, obj):
    try: 
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Dupped the object at {filepath}")
            
    except Exception as e :
        raise CustomException(e, sys)
    
def load_object(filepath):
    try:
        with open(filepath, 'rb') as file_obj:
            return dill.load(file_obj) 
    except Exception as e:
        CustomException(e, sys)

def evaluate_models(x_tr, y_tr, x_tst, y_tst, models, params : dict | None):
    try : 
        report = {}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            # Hyper parameter tuning
            if params :
                para = params[model_name]
            
                # Applying Grid Search CV
                gs = GridSearchCV(model, para, cv = 3, n_jobs=-1)
                gs.fit(x_tr, y_tr)
                logging.info(f"Completed Grid Search for {model_name}")
                model.set_params(**gs.best_params_)
        
            model.fit(x_tr, y_tr)

            # making the predections
            # y_train_pred = model.predict(x_tr)
            y_test_pred = model.predict(x_tst)

            # train_score = r2_score(y_tr, y_train_pred)
            test_score = r2_score(y_tst, y_test_pred)
            report[model_name] = test_score

        logging.info(f"Got the final test score")
        return report
    except Exception as e:
        raise CustomException(e, sys)