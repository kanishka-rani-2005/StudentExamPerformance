import os
import sys
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as f:
            dill.dump(obj,f)

    except Exception as e:
        ce=CustomException(e,sys)
        logging.error(ce)    



def evaluate_model(xtrain,ytrain,xtest,ytest,models,params,cv=3,n_jobs=3,verbose=1,refit=True):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]


            gs=GridSearchCV(model,param,cv=cv,n_jobs=n_jobs,verbose=verbose,refit=refit)
            gs.fit(xtrain,ytrain)

            model.set_params(**gs.best_params_)
            model.fit(xtrain,ytrain)

            logging.info(f'{model} fit with its best parameters.')

            # y_train_pred=model.predict(xtrain)
            y_test_pred=model.predict(xtest)

            # train_model_score=r2_score(ytrain,y_train_pred)
            test_model_score=r2_score(ytest,y_test_pred)

            report[list(models.keys())[i]]=test_model_score

        return report
    
    except Exception as e:
        ce=CustomException(e,sys)
        logging.error(ce)



def load_object(file_path):
    try:
        with open (file_path,'rb') as f:
            return dill.load(f)
    except Exception as e:
        return CustomException(e,sys)
    
    