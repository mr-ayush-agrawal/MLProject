import os
import sys
from dataclasses import dataclass

from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_cofig = ModelTrainerConfig()
    
    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting the arrays")
            x_tr, y_tr, x_tst, y_tst = (
                train_arr[:,:-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:,-1]
            )

            # Mkaing dictonary of the models
            models = {
                'Liner Regression' : LinearRegression(),
                'KNN Regressor'    : KNeighborsRegressor(),
                'Decision Tree'    : DecisionTreeRegressor(),
                'Random Forest'    : RandomForestRegressor(),
                'Gradient Boost'   : GradientBoostingRegressor(),
                'XG Boost'         : XGBRegressor(),
                'Ada Boost'        : AdaBoostRegressor()
            }

            models_report : dict = evaluate_models(x_tr, y_tr, x_tst, y_tst, models)

            # Getting the best models from the dict
            best_model_score = max(sorted(models_report.values()))
            best_model_name = list(models_report.keys())[list(models_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            
            if best_model_score < 0.6 :
                logging.warning("No best model found")
                raise CustomException("No best model found", sys)

            logging.info(f'Best model found on both training and testing datasets')

            # Saving the best model
            save_object(
                filepath=self.model_trainer_cofig.trained_model_path,
                obj=best_model
            )
            logging.info("Saved the best model")

            pred = best_model.predict(x_tst)
            r2s = r2_score(y_tst, pred)

            return r2s  

        except Exception as e:
            raise CustomException(e, sys)