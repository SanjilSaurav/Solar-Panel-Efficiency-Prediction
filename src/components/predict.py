import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, orig_features):
        try:
            features = orig_features.copy()
            drop_columns = ['id', 'module_temperature', 'pressure', 'wind_speed']
            features.drop(drop_columns, axis=1, inplace=True)
            features['humidity'] = pd.to_numeric(features['humidity'], errors='coerce')
            features[['installation_type','error_code']] = features[['installation_type','error_code']].fillna('NA')
            model_path=("artifacts\model.pkl")
            preprocessor_path = ("artifacts\preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            print("Data preprocessed Successfully")
            preds = model.predict(data_scaled)
            print("Prediction Completed")
            return preds

        except Exception as e:
            raise CustomException(e, sys)
        
