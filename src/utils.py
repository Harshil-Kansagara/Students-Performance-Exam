
import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

class Utils:
    def save_object(self, file_path, obj):
        try:
            logging.info(f"Saving object file at location: {file_path}...")
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate_models(self, X_train, y_train, X_test, y_test, models, model_params):
        try:
            report = {}
            for i in range(len(list(models))):
                model = list(models.values())[i]
                param = model_params[list(models.keys())[i]]
                
                gs = GridSearchCV(model, param, cv=3)
                gs.fit(X_train, y_train)
                
                logging.info(f"Started training for model {list(models.keys())[i]}...")
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)
                logging.info(f"Completed training for model {list(models.keys())[i]}.")
                
                logging.info(f"Started prediction for model {list(models.keys())[i]}...")
                pred = model.predict(X_test)
                logging.info(f"Completed prediction for model {list(models.keys())[i]}")
                
                test_model_score = r2_score(y_test, pred)
                logging.info(f"r2 score for model {list(models.keys())[i]} is {test_model_score}")
                report[list(models.keys())[i]] = test_model_score
                
            return report
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_object(self, file_path):
        try:
            with open(file_path, "rb") as file_obj:
                return pickle.load(file_obj)

        except Exception as e:
            raise CustomException(e, sys)