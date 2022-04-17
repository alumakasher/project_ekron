import joblib
import pandas as pd
import train


def predict(text):
    my_model = joblib.load('models/svm.pkl')
    res = my_model.predict([text])
    return res[0]

