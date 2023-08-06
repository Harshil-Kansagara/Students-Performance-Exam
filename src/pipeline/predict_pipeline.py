import sys, os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import Utils

class PredictPipeline:
    def __init__(self) -> None:
        pass
    
    def predict(self, features):
        try:
            model_path = os.path.join('../artifacts', 'model.pkl')
            preprocessor_path=os.path.join('../artifacts','preprocessor.pkl')
            utils = Utils()
            logging.info(f"Loading {model_path} pickle file...")
            model= utils.load_object(file_path=model_path)
            logging.info(f"Loading {preprocessor_path} pickle file...")
            preprocessor = utils.load_object(file_path=preprocessor_path)
            logging.info("Preprocessing started for new features...")
            data_scaled=preprocessor.transform(features)
            logging.info("Preprocessing completed for new features.")
            logging.info("Prediction started for new features...")
            preds=model.predict(data_scaled)
            logging.info("Prediction completed for new features.")
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender:str, race_ethnicity:str, parental_level_of_education:str, lunch:str, 
                 test_preparation_course:str, reading_score:int, writing_score:int) -> None:
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def convert_raw_data_to_data_frame(self):
        logging.info("Converting raw data to data frame..")
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e: 
            raise CustomException(e, sys)