#!/usr/bin/env python

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import PolynomialFeatures, StandardScaler

from train_lasso_model import train_lasso_model
from train_OLS_model import train_OLS_model
import testing_combos as combo

# Load known and unknown data
X_known = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_unknown = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_known = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]


# Set max polynomial order to test
# Given the number of features, a power of greater than three will result in more variables than we have data points. 
# For this reason, three was chosen as the maximum
max_order = 3

# Set random_state so that each model candidate uses the same folds
random_state = 2

# Select R2 as scoring metric
scoring = 'r2'

# Split data into training and test data (3:1 ratio)
X_known_train_split , X_known_test_split, y_known_train_split, y_known_test_split = train_test_split(
                        X_known, y_known, test_size=0.25, random_state=random_state)


# select and train OLS model
OLS_model_results = train_OLS_model(combo.possible_combinations, X_known_train_split, y_known_train_split, X_known_test_split, y_known_test_split, 
                        scoring, max_order, random_state)

OLS_estimator = OLS_model_results["estimator"]
OLS_estimator_test_score = OLS_model_results["test_score"]
OLS_estimator_features = tuple(OLS_model_results["features"])
OLS_estimator_subset = OLS_model_results["subset"]
OLS_estimator_order = OLS_model_results["order"]

# select and train lasso model
lasso_model_results = train_lasso_model(combo.possible_combinations, X_known_train_split, y_known_train_split, X_known_test_split, y_known_test_split, 
                        scoring, max_order, random_state)

lasso_estimator = lasso_model_results["estimator"]
lasso_estimator_test_score = lasso_model_results["test_score"]
lasso_estimator_features = tuple(lasso_model_results["features"])
lasso_estimator_order = lasso_model_results["order"]
lasso_estimator_alpha = lasso_model_results["alpha"]



# Output best models for OLS and Lasso

print("\n\nBest Results for OLS and Lasso:\n")
print("\nOLS Best model: features: ", OLS_estimator_features, ", subset = ", OLS_estimator_subset, ",  order = ", OLS_estimator_order, ", testscore = ", OLS_estimator_test_score)
print("\nLasso Best model: features: ", lasso_estimator_features, ", alpha = ",  lasso_estimator_alpha, ",  order = ", lasso_estimator_order, ", testscore = ", lasso_estimator_test_score)

# compare results

if (lasso_estimator_test_score > OLS_estimator_test_score):
  print("\nLasso is expected to outperform OLS")
elif (OLS_estimator_test_score > lasso_estimator_test_score):
  print("\nOLS is expected to outperform Lasso")
else:
  print("\nOLS and Lasso are the same?! Best to assume the code is bugged")



# Make predictions for OLS

## Apply Combination function

X_known_OLS_format = combo.possible_combinations[OLS_estimator_features](X_known)
X_unknown_OLS_format = combo.possible_combinations[OLS_estimator_features](X_unknown)

## Extract subset

X_known_OLS_format  = X_known_OLS_format[:,OLS_estimator_subset]
X_unknown_OLS_format = X_unknown_OLS_format[:,OLS_estimator_subset]

## Convert known and unknown data to correct format
poly = PolynomialFeatures(OLS_estimator_order, include_bias=False)
X_known_OLS_format = poly.fit_transform(X_known_OLS_format)
X_unknown_OLS_format = poly.transform(X_unknown_OLS_format)

scaler = StandardScaler() 
X_known_OLS_format = scaler.fit_transform(X_known_OLS_format)
X_unknown_OLS_format  = scaler.transform(X_unknown_OLS_format)

## Train on full known data
OLS_estimator.fit(X_known_OLS_format,y_known)
y_pred_OLS = OLS_estimator.predict(X_unknown_OLS_format)



# Make predictions for lasso


## Apply Combination function

X_known_lasso_format = combo.possible_combinations[lasso_estimator_features](X_known)
X_unknown_lasso_format = combo.possible_combinations[lasso_estimator_features](X_unknown)


## Convert known and unknown data to correct format
poly = PolynomialFeatures(lasso_estimator_order, include_bias=False)
X_known_lasso_format = poly.fit_transform(X_known_lasso_format)
X_unknown_lasso_format = poly.transform(X_unknown_lasso_format)

scaler = StandardScaler() 
X_known_lasso_format = scaler.fit_transform(X_known_lasso_format)
X_unknown_lasso_format  = scaler.transform(X_unknown_lasso_format)

## Train on full known data
lasso_estimator.fit(X_known_lasso_format,y_known)
y_pred_lasso = lasso_estimator.predict(X_unknown_lasso_format)


# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,PRP"
n_points = X_unknown.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)

y_pred_pp[:, 1] = y_pred_OLS
np.savetxt('OLS_predictions.csv', y_pred_pp, fmt='%d,%f', delimiter=",",
           header=test_header, comments="")

y_pred_pp[:, 1] = y_pred_lasso
np.savetxt('Lasso_predictions.csv', y_pred_pp, fmt='%d,%f', delimiter=",",
           header=test_header, comments="")