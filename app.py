import sys
import pandas as pd

from src.components.predict import PredictPipeline

if __name__=="__main__":
    pred_df = pd.read_csv('test.csv')
    predict_inst = PredictPipeline()
    print("Mid prediction")
    results=predict_inst.predict(pred_df)
    print("Finished Prediction")
    print(pred_df.columns)
    id_pred = pred_df['id']
    pred_dct = {'id': id_pred, 'efficiency':results}
    sub_result = pd.DataFrame(pred_dct)
    sub_result.to_csv('submission_result.csv', index=False)
    print(results)