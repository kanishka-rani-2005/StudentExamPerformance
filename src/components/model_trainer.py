import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object ,evaluate_model


@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split Training and Test input Data.')
            
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info('Data get splitted properly.')

            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                'XGBRegressor':XGBRegressor(),
                'AdaBoostRegressor':AdaBoostRegressor(),
                'CatBoostRegressor':CatBoostRegressor(),
                'SVR':SVR(),
                'GradientBoostRegressor':GradientBoostingRegressor()
            }

            logging.info('All the models we will try .')

            r2_list=[]
            model_report:dict=evaluate_model(xtrain=x_train,ytrain=y_train,xtest=x_test,ytest=y_test,models=models)


            logging.info('Get model report for each model.')

            # get best model score from dict 
            best_model_score=max(sorted(model_report.values()))

            # to get best model name from dict

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                ce= CustomException('No Best model found')
                logging.error(ce)
            
            logging.info('Best model found on both training and testing dataset.')
            logging.info(f"Best model name is {best_model_name}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info('Model saved succesfully.')

            predicted=best_model.predict(x_test)
            score=r2_score(y_test,predicted)

            logging.info(f'R2 Score is {score}')
            return score

        except Exception as e:
            ce=CustomException(e,sys)
            logging.error(ce)

             
