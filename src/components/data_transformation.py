import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self, num_col, cat_col):
        try:
            numerical_columns = num_col
            categorical_columns = cat_col

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", KNNImputer()),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            drop_columns = ['id', 'module_temperature', 'pressure', 'wind_speed']
            train_df.drop(drop_columns, axis=1, inplace=True)
            test_df.drop(drop_columns, axis=1, inplace=True)
            train_df['humidity'] = pd.to_numeric(train_df['humidity'], errors='coerce')
            test_df['humidity'] = pd.to_numeric(test_df['humidity'], errors='coerce')
            numerical_columns = [clm for clm in train_df.columns if train_df[clm].dtype != 'object']
            categorical_columns = [clm for clm in train_df.columns if train_df[clm].dtype == 'object']
            numerical_columns.remove('efficiency')

            train_df[categorical_columns] = train_df[categorical_columns].fillna('NA')
            test_df[categorical_columns] = test_df[categorical_columns].fillna('NA')
            
            preprocessing_obj = self.get_data_transformer_object(numerical_columns, categorical_columns)

            target_column_name = 'efficiency'
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            print(input_feature_train_df.columns)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            print('1............................................')
            print(input_feature_train_df.columns)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            print('1.5............................')
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            print('2.................................................')
            print(np.array(target_feature_train_df).shape)
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            print('3....................................................')
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            print('4..........................................................')
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            print('5...............................................')
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)