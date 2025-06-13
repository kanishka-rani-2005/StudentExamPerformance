import os
import sys
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as f:
            dill.dump(obj,f)

    except Exception as e:
        ce=CustomException(e,sys)
        logging.error(ce)    



def evaluate_model(xtrain,ytrain,xtest,ytest,models):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]

            model.fit(xtrain,ytrain)

            # y_train_pred=model.predict(xtrain)
            y_test_pred=model.predict(xtest)


            # train_model_score=r2_score(ytrain,y_train_pred)
            test_model_score=r2_score(ytest,y_test_pred)

            report[list(models.keys())[i]]=test_model_score

        return report
    
    except Exception as e:
        ce=CustomException(e,sys)
        logging.error(ce)
