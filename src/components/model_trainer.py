import os 
import sys

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import evaluate_models,save_object
@dataclass
class ModelTrainerConfig:
   trained_model_path=os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train,y_train,X_test,y_test=(
            train_array[:,:-1],
            train_array[:,-1],
            test_array[:,:-1],
            test_array[:,-1],
            )
            models={
                'Linear Regression' : LinearRegression(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Random Forest" : RandomForestRegressor(),
                'KNeighobors Regressor' : KNeighborsRegressor(),
                "AdaBoost Regressor"   : AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor"      : XGBRegressor(verbosity=0,
                                                    random_state=42),
                "CatBoosting Regressor"      : CatBoostRegressor(verbose=False)
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Linear Regression': {},
                'KNeighobors Regressor': {},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_params:dict=evaluate_models(X_train=X_train,y_train=y_train,
                                                X_test=X_test,y_test=y_test,models=models,param=params)
            best_model_score=max(sorted(model_params.values()))
            best_model_name=list(model_params.keys())[
                list(model_params.values()).index(best_model_score)

            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")


            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            return best_model_score,best_model_name

        except Exception as e:
            raise CustomException(e,sys)
        
        

        


