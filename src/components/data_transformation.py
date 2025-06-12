import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import os

from src.logger import logging
from src.exception import CustomException
import pickle

from src.utils import save_object


@dataclass
class DataTransformConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')



class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformConfig()


    def get_data_transformer_object(self):
    
        '''
        Function Responsible for data transformation
        '''


        try:
            
            num_cols=['writing_score','reading_score']
            cat_cols=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False)),
                ]

            )
            logging.info('Numerical columns Standard scaling completed.')

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info('Categorical columns encoding completed.')


            logging.info(f"Categorical Columns {cat_cols}")
            logging.info(f"Numerical Columns {num_cols}")

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_cols),
                    ('cat_pipeline',cat_pipeline,cat_cols)
                ]
            )

            logging.info('Column Transformer Done.')

            return preprocessor
        
        except Exception as e:

            ce=CustomException(e,sys)
            logging.error('ERROR!!!')
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read Train and Test Data Completed.')
            logging.info('Obtaining preprocessing object .')

            preprocessor_obj=self.get_data_transformer_object()

            target_column_name='math_score'

            input_feature_train_df=train_df.drop(target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]
           
            input_feature_test_df=test_df.drop(target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info('Applying preprocessing object on training df and testing df.')

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f"Saved Preprocessing object.")

# in utils write save_object function
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info('Preprocessor Saved.')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            ce= CustomException(e,sys)
            logging.error(ce)