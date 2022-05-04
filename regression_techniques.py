import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import seaborn as sns

def preprocessing(df):
    df = df.drop([220,270,312],axis=0) #drop outliers
    df = df.reset_index()
    df = df.drop(['No', 'X1 transaction date', 'X5 latitude', 'X6 longitude'], axis=1) # not relevant
    df = df.dropna()

    list_indices = [i for i in range(df.shape[0])]
    random.shuffle(list_indices)

    d_train_indices = list_indices[:int((len(list_indices)/3)*2)]
    d_test_indices = list_indices[int((len(list_indices)/3)*2):]

    house_price = df['Y house price of unit area']
    d_train_targets = house_price[d_train_indices]
    d_test_targets = house_price[d_test_indices]

    data = df.drop(['Y house price of unit area'], axis=1)
    poly = PolynomialFeatures(2)
    extract = poly.fit_transform(data)

    d_train = extract[d_train_indices]
    d_test = extract[d_test_indices]

    #standardization
    scaler = StandardScaler()
    scaler.fit(d_train)
    d_train = scaler.transform(d_train)
    d_test = scaler.transform(d_test)
    return (df, d_train, d_train_targets, d_test, d_test_targets)

def linear_no_pfe(d_train, d_train_targets, d_test, d_test_targets): 
    
    reg = LinearRegression().fit(d_train, d_train_targets)
    #print(reg.score(d_train, train_target))
    test_score = reg.score(d_test, d_test_targets)
    pred = reg.predict(d_test) #predict


    scores = cross_val_score(reg, d_train, d_train_targets, scoring='r2', cv=5)
    return(np.mean(scores), test_score, pred)

#LINEAR WITH PFE 
def linear_w_pfe(d_train, d_train_targets, d_test, d_test_targets): 
    reg = LinearRegression().fit(d_train, d_train_targets)
    train_score = reg.score(d_train, d_train_targets)
    test_score = reg.score(d_test, d_test_targets)
    pred = reg.predict(d_test) #predict

    #print(train_score)
    #print(test_score)

    scores = cross_val_score(reg, d_train, d_train_targets, scoring='r2', cv=5)
    return (np.mean(scores), test_score, pred)

def ridge(d_train, d_train_targets, d_test, d_test_targets): 
    #RIDGECV
    ridgecv = RidgeCV(cv=5)
    ridgecv.fit(d_train, d_train_targets)
    test_score = ridgecv.score(d_test, d_test_targets)
    pred = ridgecv.predict(d_test) #predict

    scores = cross_val_score(ridgecv, d_train, d_train_targets, scoring='r2', cv=5)
    return (np.mean(scores), test_score, pred)

def lasso(d_train, d_train_targets, d_test, d_test_targets):
    #LASSOCV
    reg = LassoCV(cv=5, random_state=0)
    reg.fit(d_train, d_train_targets)
    test_score = reg.score(d_test, d_test_targets)

    pred = reg.predict(d_test)

    scores = cross_val_score(reg, d_train, d_train_targets, scoring='r2', cv=5)
    return (np.mean(scores), test_score, pred)

#VISUALIZATION
def correlation_matrix(df):  
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.show()

def pairplot(df):   
    sns.pairplot(df)
    plt.show()

def scatter(title, x_label, y_label, pred, d_test_targets): 
    plt.scatter(d_test_targets,pred)
    m,b = np.polyfit(d_test_targets,pred, 1)
    plt.plot(d_test_targets, m*d_test_targets+ b)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def boxplot(df): 
    #boxplot to find outliers, find outliers
    plt.boxplot(df["Y house price of unit area"])
    plt.show()


def main(): 
    #df, d_train, d_train_targets, d_test, d_test_targets
    df = pd.read_csv("Real estate.csv")
    #make boxplot to check for outliers 
    #boxplot(df)
    #see we need to remove outliers 
    preprocessed = preprocessing(df)
    df = preprocessed[0]
    d_train = preprocessed[1]
    d_train_targets = preprocessed[2]
    d_test = preprocessed[3]
    d_test_targets = preprocessed[4]
    #call the models 

    lin_no_pfe = linear_no_pfe(d_train, d_train_targets, d_test, d_test_targets)
    pred = lin_no_pfe[2]
    print("Linear Regression without PFE Mean Score: ", lin_no_pfe[0], "Linear Regression with PFE Mean Score: ", lin_no_pfe[1])
    scatter("Linear Regression without PFE Prediction vs. Actual", "Linear Prediction", "Actual", pred, d_test_targets)

    lin_w_pfe = linear_w_pfe(d_train, d_train_targets, d_test, d_test_targets)
    pred = lin_w_pfe[2]
    print("Linear Regression with PFE Mean Score: ", lin_w_pfe[0], "Linear Regression with PFE Mean Score: ", lin_w_pfe[1])
    scatter("Linear Regression with PFE Prediction vs. Actual", "Linear Prediction", "Actual", pred, d_test_targets)

    rid = ridge(d_train, d_train_targets, d_test, d_test_targets)
    print("Ridge Mean Score: ", rid[0], "Ridge Test Score: ", rid[1])
    pred = rid[2]
    scatter("Ridge Regression Prediction vs. Actual", "Ridge Prediction", "Actual", pred, d_test_targets)

    las= lasso(d_train, d_train_targets, d_test, d_test_targets)
    print("Lasso Mean Score: ", las[0], "Lasso Test Score: ", las[1])
    pred = las[2]
    scatter("Lasso Regression Prediction vs. Actual", "Lasso Prediction", "Actual", pred, d_test_targets)

main()





    
