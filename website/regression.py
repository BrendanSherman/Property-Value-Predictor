import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import RidgeCV


def ridge_pred(usr_input):
    df = pd.read_csv("Real estate.csv")
    df = df.drop(['No'], axis=1)
    df = df.dropna()

    house_price = df['Y house price of unit area']
    data = df.drop(['Y house price of unit area'], axis=1)

    poly = PolynomialFeatures(2)
    poly = poly.fit(data)
    poly2 = PolynomialFeatures(2)
    poly2 = poly2.fit(data)

    data = poly.transform(data)
    usr_input = poly2.transform(usr_input)

    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    usr_input = scaler.transform(usr_input)

    #Use ridge model to predict target value for user input  
    ridgecv = RidgeCV(cv=5)
    ridgecv.fit(data, house_price)
    ridge_reg = Ridge(alpha = ridgecv.alpha_)
    ridge_reg.fit(data, house_price)
    prediction = ridge_reg.predict(usr_input)
    return prediction







    
