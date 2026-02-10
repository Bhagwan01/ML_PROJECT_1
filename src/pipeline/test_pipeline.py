import sys
import numpy as np
import pandas as pd
from src.utils import load_object
from src.exception import CustomException
import os

class PredictPipeline:
    def __init__(self):
        pass
    def prediction(self,features):
        try:
            model_path=os.path.join('artifact','model.pkl')
            preprocessor_path=os.path.join('artifact','preprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            processed_data=preprocessor.transform(features)
            predictions=model.predict(processed_data)
            return predictions
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,gender:str,race_ethnicity:str,parental_level_of_education:str,lunch:str,
                 test_preparation_course:str,reading_score:int,writing_score:int):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_dataFrame_from_data(self):
        try:
            custom_data_dict={
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]

            }
            df=pd.DataFrame(custom_data_dict)
            return df
        except Exception as e:
            raise CustomException(e,sys)

