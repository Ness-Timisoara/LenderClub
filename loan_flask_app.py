from flask import Flask, Request, request, render_template
from werkzeug.datastructures import ImmutableOrderedMultiDict
from loan_predict import predict
import pandas as pd
import pickle
import os


class OrderedRequest(Request):
    parameter_storage_class = ImmutableOrderedMultiDict


class MyFlask(Flask):
    request_class = OrderedRequest


app = MyFlask(__name__, static_folder="public", template_folder="views")


@app.route("/")
def homepage():
    return render_template("index.html")


@app.route("/api/predict")
def loan_risk_predict():
    assert isinstance(request.args, ImmutableOrderedMultiDict)
    print('CALLING PREDICT==============>>>>>>>', request.args)
    print('type(PREDICT)==============>>>>>>>', type(request.args))
    # pickle.dump(request.args, open(os.path.join("data","evaluation_data.pkl"), "wb"))
    prediction = predict(request.args)

    print('prediction = ',prediction)


    if pd.isna(prediction):
        return {"error": "There's something wrong with your input."}, 400

    if prediction < 0:
        prediction = 0
    elif prediction > 1:
        prediction = 1

    description = f"This loan is predicted to recover {round(prediction * 100, 1)}% of its expected return."

    return {"value": str(prediction), "description": description}


if __name__ == "__main__":
    app.run()
