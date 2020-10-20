import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from loan_predict_preprocess import run_dfmapper

from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout


def predict(arguments):
    try:
        print('#######Loading input data into DF')
        data = pd.DataFrame(arguments, index=[0])

        for col_name in data:
            data[col_name] = pd.to_numeric(data[col_name], errors="ignore")

        inverse_columns = [col_name for col_name in data if "inv_" in col_name]

        print('#######inverse_columns = ',inverse_columns)

        def invert(value):
            if type(value) is str:
                return 0
            else:
                return 1 / 1 if value == 0 else 1 / value

        for col_name in inverse_columns:
            data[col_name] = data[col_name].map(invert)

        for joint_col, indiv_col in zip(["annual_inc_joint", "dti_joint"], ["annual_inc", "dti"]):
            data[joint_col] = [
                indiv_val if type(joint_val) is str else joint_val
                for joint_val, indiv_val in zip(data[joint_col], data[indiv_col])
            ]

        print('#######inverse_columns = ', inverse_columns)

        print('CALLING MODEL WITH DATA==============>>>>>>>',data)
        ###transformer = joblib.load("models/data_transformer.joblib")
        print('CALLING DF MAPPER==============>>>>>>>')
        X77 = run_dfmapper(data)
        print('MODEL run_dfmapper returned ==============>>>>>>>', X77)
        print('MODEL run_dfmapper returned SHAPE==============>>>>>>>', X77.shape)
        model = load_model("models/loan_risk_model")
        print('loan_risk_model-MODEL LOADED==============>>>>>>>')
        result = model(X77).numpy()[0][0]

        return result

    except:
        return np.nan


