from tensorflow import keras
import argparse
import os
import pickle
from werkzeug.datastructures import ImmutableOrderedMultiDict
from loan_predict import predict


def evaluate():
    evaluation_data = pickle.load(open(os.path.join("data","evaluation_data.pkl"), "rb"))
    assert isinstance(evaluation_data, ImmutableOrderedMultiDict)
    print('CALLING PREDICT==============>>>>>>>', evaluation_data)
    print('type(PREDICT)==============>>>>>>>', type(evaluation_data))
    prediction = predict(evaluation_data)
    print("Prediction = ",prediction)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubeflow  training script')
    parser.add_argument('--loadtype',help='use eval as the keyword')
    args = parser.parse_args()
    evaluate()