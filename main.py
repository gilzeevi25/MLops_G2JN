import numpy as np
import pandas as pd
from pipeline import G2JN_Pipeline
def load_boston():
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    boston_dataset = pd.read_csv("data/housing.csv", header=None, delimiter=r"\s+", names=column_names)
    X_bos, y_bos = boston_dataset.iloc[:,:-1], boston_dataset.iloc[:,-1]
    name = "Boston-Housing"
    return  X_bos, y_bos, name


def load_motors():
    motors_dataset = pd.read_csv('data/freMTPL2freq.csv')
    # Convert Categorical features to Numerical Features
    for col in ['Area','VehBrand','VehGas','Region']:
        d = {}
        for i, val in enumerate(motors_dataset[col].unique()):
            d[val] = i + 1
        motors_dataset[col] = motors_dataset[col].apply(lambda x: d[x])
    # Calculate Frequency
    motors_dataset['Frequency'] = motors_dataset['ClaimNb'] / motors_dataset['Exposure']
    X_mot, y_mot = motors_dataset.drop(['IDpol', 'ClaimNb', 'Exposure', 'Frequency'],axis=1), motors_dataset['Frequency']
    name = "French-Motor-claims"
    return X_mot, y_mot,name

if __name__ == "__main__":
    X_bos, y_bos, name = load_boston()
    print("\n--------------------------------")
    print("--------------------------------")
    print("--------------------------------\n")
    _ = G2JN_Pipeline(X_bos, y_bos, name).fit_transform()
    print("\n--------------------------------")
    print("--------------------------------")
    print("--------------------------------\n")
    X_mot, y_mot, name = load_motors()
    _ = G2JN_Pipeline(X_mot, y_mot, name ).fit_transform()

