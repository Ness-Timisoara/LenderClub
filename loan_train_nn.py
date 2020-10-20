from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
import joblib
import argparse
import os
import pickle
import numpy as np


def run_pipeline(
    data, onehot_cols, ordinal_cols, batch_size, validate=True,
):
    X = data.drop(columns=["fraction_recovered"])
    y = data["fraction_recovered"]
    X_train, X_valid, y_train, y_valid = (
        train_test_split(X, y, test_size=0.2, random_state=0)
        if validate
        else (X, None, y, None)
    )

    transformer = DataFrameMapper(
        [
            (onehot_cols, OneHotEncoder(drop="if_binary")),
            (
                list(ordinal_cols.keys()),
                OrdinalEncoder(categories=list(ordinal_cols.values())),
            ),
        ],
        default=StandardScaler(),
    )

    X_train = transformer.fit_transform(X_train)
    X_valid = transformer.transform(X_valid) if validate else None

    input_nodes = X_train.shape[1]
    output_nodes = 1

    model = Sequential()
    model.add(Input((input_nodes,)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3, seed=0))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.3, seed=1))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.3, seed=2))
    model.add(Dense(output_nodes))
    model.compile(optimizer="adam", loss="mean_squared_error")

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=100,
        validation_data=(X_valid, y_valid) if validate else None,
        verbose=1,
    )

    return history.history, model, transformer

def add_trail_slash(s):
    # if not s.startswith('/'):
    #     s = '/'+s
    if not s.endswith('/'):
        s = s+'/'
    return s

def remove_trail_slash(s):
    # if not s.startswith('/'):
    #     s = '/'+s
    if s.endswith('/'):
        s = s[:-1]
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubeflow training script')
    parser.add_argument('--trainmode', help='train retrain')
    parser.add_argument('--input_path', help='path where you have saved the outputs from the preprocessing')
    parser.add_argument('--output_dir', help='path where you want to save the outputs from the training')
    args = parser.parse_args()

    input_dir = args.input_path
    if args.input_path is None:
        input_dir='data'

    output_dir = args.output_dir
    if args.output_dir is None:
        output_dir='models'

    input_dir=add_trail_slash(input_dir)
    print('output_dir=',output_dir)


    if args.trainmode == 'train':
        pkl_file_loan         = os.path.join(input_dir,'train_load_1.pickle')
        pkl_file_onehot_cols  = os.path.join(input_dir,'train_onehot_cols.pickle')
        pkl_file_ordinal_cols = os.path.join(input_dir,'train_ordinal_cols.pickle')

        file1 = open(pkl_file_loan, 'rb')
        loans_1 = pickle.load(file1)
        file1.close()

        file2 = open(pkl_file_onehot_cols, 'rb')
        onehot_cols = pickle.load(file2)
        file2.close()

        file3 = open(pkl_file_ordinal_cols, 'rb')
        ordinal_cols = pickle.load(file3)
        file3.close()

        history, final_model, final_transformer = run_pipeline(loans_1,onehot_cols,ordinal_cols,batch_size=128,validate=False,)
        final_model.save(os.path.join(output_dir,"loan_risk_model"))
        joblib.dump(final_transformer, os.path.join(output_dir,"data_transformer.joblib"))

        # sns.lineplot(x=range(1, 101), y=history["loss"], label="loss")
        # sns.lineplot(x=range(1, 101), y=history["val_loss"], label="val_loss")
        # plt.xlabel("epoch")
        # plt.title("Model 1 loss metrics during training")
        # ###plt.show()
        # plt.savefig('reports/LossMetric.png')


