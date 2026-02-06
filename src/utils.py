import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try :
        model_params={}
        for i in range(len(models)):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            train_score=r2_score(y_train,y_train_pred)
            test_score=r2_score(y_test,y_test_pred)
            
            model_params[list(models.keys())[i]] = test_score
            
        
        return model_params
    except Exception as e :
        raise CustomException(e,sys)

