import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import RidgeCV

#trains and utilizes ridge regression model to return a predicted sale price given user input
def ridge_pred(usr_input):
    df = pd.read_csv("Real estate.csv")
    df = df.drop(['No'], axis=1) #index column, unused
    df = df.dropna() #drop rows with missing entries

    #locate and isolate target attribute (sale price)
    sale_price = df['Y house price of unit area'] 
    data = df.drop(['Y house price of unit area'], axis=1)


    #creates polynomial features for data and user input
    poly = PolynomialFeatures(2)
    poly = poly.fit(data)
    poly2 = PolynomialFeatures(2)
    poly2 = poly2.fit(data)
    data = poly.transform(data)
    usr_input = poly2.transform(usr_input)

    #Standardizes data and input
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    usr_input = scaler.transform(usr_input)

    #Use ridge model to predict target value for user input  
    ridgecv = RidgeCV(cv=5)
    ridgecv.fit(data, sale_price)
    ridge_model = Ridge(alpha = ridgecv.alpha_)
    ridge_model.fit(data, sale_price)
    prediction = ridge_model.predict(usr_input)

    return prediction







    
