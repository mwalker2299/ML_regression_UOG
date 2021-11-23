#!/usr/bin/env python

import numpy as np

from sklearn.metrics import get_scorer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_validate
from itertools import chain, combinations
from operator import itemgetter

import warnings
warnings.filterwarnings("ignore")

# From https://docs.python.org/3/library/itertools.html#itertools.chain
# Constructs a powerset from the given list. This will be used to evaluate
# each possible subset of features
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def find_optimum_model(feature_combos, X_train, y_train, cv, scoring = 'neg_mean_squared_error', max_order=3):
    
  results_list = []
  combo_counter = 0

  print("Start selecting OLS model: ")

  for descriptor, combination in feature_combos.items():
    descriptor = np.array(descriptor)
    subset_counter = 0
    combo_counter+=1
    print ("combo:", combo_counter,"/",len(feature_combos.keys()), flush=True) # Provide indication of combo progress

    X_train_combo = combination(X_train)

    all_subsets = np.array(list(powerset(np.arange(X_train_combo.shape[1])))[1:])

    for subset in all_subsets:
      subset_counter+=1
      print ("subset:", subset_counter,"/",len(all_subsets), flush=True) # Provide indication of subset progress

      subset = np.array(subset)
      X_train_combo_subset = X_train_combo[:,subset]
    
      for i in np.arange(max_order)+1:

        model_results = {}

        # Generate polynomial for current feature combo. 
        # No need for an intercept as this will be added by the Lasso algorithm
        poly = PolynomialFeatures(i, include_bias=False)
        X_train_combo_subset_poly = poly.fit_transform(X_train_combo_subset)

        # Scale data around its mean before training
        scaler = StandardScaler() 
        X_train_combo_subset_poly_scaled = scaler.fit_transform(X_train_combo_subset_poly)

        regmodel = LinearRegression()
      
        CV_results = cross_validate(
            regmodel, X_train_combo_subset_poly_scaled, y_train, scoring=scoring, cv=cv)

        model_results["features"]   = descriptor
        model_results["subset"]     = subset
        model_results["order"]      = i
        model_results["estimator"]  = regmodel

        #Calculate Mean R-squared
        model_results["validation_score"] = np.mean(CV_results["test_score"])


        results_list.append(model_results)
  
  # Sort results based on best validation data score
  sorted_results = sorted(results_list, key=itemgetter('validation_score'))

  return sorted_results

# Selects and trains the optimum Lasso model.
# Examines every subset of all defined features combos at poly powers 1,2 and 3
def train_OLS_model (feature_combos, X_train,y_train, X_test, y_test, cv, scoring, max_order):

  results = find_optimum_model(feature_combos, X_train, y_train, cv=cv, scoring=scoring, max_order=max_order)

  print("\n\n OLS results:\n")
  for result in results:
    print("features: ", result["features"][result["subset"]], ",  order: ", result["order"], ", validation score: ", np.sqrt(abs(result["validation_score"])))

  # Retrieve estimator with optimum test score.
  best_model_results = results[-1]

  # We need to calculate a test score for comparison before returning our function

  ## Apply Combination function

  combination = feature_combos[tuple(best_model_results["features"])]

  X_train_combo = combination(X_train)
  X_test_combo = combination(X_test)

  ## Extract subset

  X_test_combo_subset  = X_test_combo[:,best_model_results["subset"]]
  X_train_combo_subset = X_train_combo[:,best_model_results["subset"]]

  ## Convert train and test data to correct format

  poly = PolynomialFeatures(best_model_results["order"], include_bias=False)
  X_train_combo_subset_poly = poly.fit_transform(X_train_combo_subset)
  X_test_combo_subset_ploy  = poly.transform(X_test_combo_subset)

  scaler = StandardScaler() 
  X_train_combo_subset_poly_scaled = scaler.fit_transform(X_train_combo_subset_poly)
  X_test_combo_subset_poly_scaled  = scaler.transform(X_test_combo_subset_ploy)

  # Calculate score of best model from CV on test data

  scorer = get_scorer(scoring)

  estimator = best_model_results["estimator"]

  estimator.fit(X_train_combo_subset_poly_scaled, y_train)
  test_score = scorer(estimator, X_test_combo_subset_poly_scaled, y_test)

  best_model_results["test_score"] = test_score



  return best_model_results

