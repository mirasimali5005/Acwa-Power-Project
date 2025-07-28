import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# reading the csv

main_file = pd.read_csv("Rw_data.csv")
# main_file

# data preprocessing not required since data is already cleaned 

# x will be the parameters im not guessing 
X = main_file.drop(columns=["next_sdi_predicted", "exact_date"])
# X

# y will be the predicted sdi
Y = main_file["next_sdi_predicted"]
# Y

# 80% train 20% test

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20, shuffle=True, stratify=None)


# choose random forest for now 
rf = RandomForestRegressor(random_state=20)
rf.fit(x_train, y_train)

y_predict = rf.predict(x_test)

# print the mse, rSquare 
# print(mean_absolute_error(y_test, y_predict)
# ,mean_squared_error(y_test, y_predict)
# ,r2_score(y_test, y_predict))

param_grid = { 'n_estimators': [100, 200, 300],
               'max_depth': [ 10, 20, 30],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4], 
               }

rf_cv = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
rf_cv.fit(x_train, y_train)

y_predict = rf_cv.predict(x_test)

print("MAE:", mean_absolute_error(y_test, y_predict))
print("MSE:", mean_squared_error(y_test, y_predict))
print("R² Score:", r2_score(y_test, y_predict))

# now to fine tune we will not see only for random forest we will see for other regression models aswell
# see the hyperparameter from the documentation and then test out atleast 3 models with diff hyperparameters
# this will be done in a for loop and using GridSearchCV

models = {
    "RandomForest": RandomForestRegressor(random_state=20),
    "GradientBoosting": GradientBoostingRegressor(random_state=20),
    "SVR": SVR()
}

param_grids = {
    "RandomForest": {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    "GradientBoosting": {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    },
    "SVR": {
        'kernel': ['rbf', 'linear'],
        'C': [1, 10],
        'epsilon': [0.01, 0.1]
    }
}

# now try using RandomSearchCV

# see the mse, rSquare for all and then decide whihc is the best model for now 

