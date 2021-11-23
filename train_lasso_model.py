#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import get_scorer
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_validate
from operator import itemgetter

import warnings

# Some of the poorer models will throw convergence warnings
# This line prevents these warnings from cluttering the output
warnings.filterwarnings("ignore")

#Plot the test, validation and training MSE scores against the range of alpha values
def plot_alpha_grid(
    validation_score, train_score, alphas_grid, chosen_alpha,
    scoring, test_score = None, filename = None):
    
    plt.figure(figsize = (8,8))
    sns.lineplot(y = validation_score, x = alphas_grid, 
                 label = 'validation_data')
    sns.lineplot(y = train_score, x = alphas_grid, 
                 label = 'training_data')
    plt.axvline(x=chosen_alpha, linestyle='--')
    if test_score is not None:
        sns.lineplot(y = test_score, x = alphas_grid, 
                     label = 'test_data')
    plt.xlabel('alpha_parameter')
    plt.ylabel(scoring)
    plt.title(filename)
    plt.legend()
    if filename is not None:
        plt.savefig(str(filename) + ".png")

# Perform repeated cross-validation to find the alpha value which produces the highest average MSE score 
def find_optimum_alpha(potential_alphas, X_train, y_train, X_test, y_test, cv, scoring = 'neg_mean_squared_error', draw_plot = True, filename = None):
    
    validation_scores = []
    train_scores = []
    model_list = []

    if X_test is not None:
        test_scores = []
        scorer = get_scorer(scoring)
    else:
        test_scores = None

    # For each alpha in the given range, perform CV
    for current_alpha in potential_alphas:

      regmodel = Lasso(alpha=current_alpha, max_iter=1000)
        
      results = cross_validate(
          regmodel, X_train, y_train, scoring=scoring, cv=cv, return_estimator=True, return_train_score=True)

      validation_scores.append(np.mean(results['test_score']))
      model_list.append(regmodel)
      train_scores.append(np.mean(results['train_score']))

      if X_test is not None:
          regmodel.fit(X_train,y_train)
          test_scores.append(scorer(regmodel, X_test, y_test))

    
    optimum_alpha = potential_alphas[np.argmax(validation_scores)]
    max_validation_score = np.max(validation_scores)
    estimator = model_list[np.argmax(validation_scores)]

    if X_test is not None:
        test_score_at_chosen_alpha = test_scores[np.argmax(validation_scores)]
    else:
        test_score_at_chosen_alpha = None

    if draw_plot:
        plot_alpha_grid(
            validation_scores, train_scores, potential_alphas, optimum_alpha, 
            scoring, test_scores, filename)

    return optimum_alpha, max_validation_score, estimator, test_score_at_chosen_alpha


# For each possible feature combination (defined in testing_combos.py),
# 
def find_optimum_model(feature_combos, X_train, y_train, X_test, y_test,
                   cv, scoring = 'neg_mean_squared_error', max_order=3):

  results_list = []
  combo_counter = 0

  print("Start selecting lasso model: ")

  for descriptor, combination in feature_combos.items():
    descriptor = np.array(descriptor)
    combo_counter+=1
    print ("combo:", combo_counter,"/",len(feature_combos.keys()), flush=True) # Provide indication of progress

    X_train_combo = combination(X_train)
    X_test_combo = combination(X_test)
    
    for i in np.arange(max_order)+1:

      results = {}

      # Generate polynomial for current feature combo. 
      # No need for an intercept as this will be added by the Lasso algorithm
      poly = PolynomialFeatures(i, include_bias=False)
      X_train_combo_poly = poly.fit_transform(X_train_combo)
      X_test_combo_ploy  = poly.transform(X_test_combo)

      # Scale data around its mean before training
      scaler = StandardScaler() 
      X_train_combo_poly_scaled = scaler.fit_transform(X_train_combo_poly)
      X_test_combo_poly_scaled  = scaler.transform(X_test_combo_ploy)

      ## start with a grid search with low level granularity and then refine.
      alpha_grid = np.linspace(-5,5,101)
      optimum_alpha, optimum_alpha_validation_score, estimator, test_score_at_optimum_alpha = find_optimum_alpha(alpha_grid, 
                                                                        X_train_combo_poly_scaled, y_train, X_test_combo_poly_scaled, y_test, 
                                                                        cv=cv, scoring=scoring, draw_plot=True, filename='plots/combo'+str(combo_counter)+'_P'+str(i)+'_R'+str(0))

      # Perform search again around optimum alpha at increased granularity
      #  The first search has a granularity of 1/10, 
      # so the second serach is bounded to +-1/10 of the initial alpha
      alpha_grid = np.linspace(optimum_alpha-0.1, optimum_alpha+0.1, 100)
      optimum_alpha, optimum_alpha_validation_score, estimator, test_score_at_optimum_alpha = find_optimum_alpha(alpha_grid, 
                                                                        X_train_combo_poly_scaled, y_train, X_test_combo_poly_scaled, y_test,
                                                                        cv=cv, scoring=scoring, draw_plot=True, filename='plots/combo'+str(combo_counter)+'_P'+str(i)+'_R'+str(1))


      # Add results to list
      results["validation_score"] = optimum_alpha_validation_score
      results["test_score"] = test_score_at_optimum_alpha
      results["features"] = descriptor
      results["order"]    = i
      results["alpha"] = optimum_alpha
      results["estimator"] = estimator


      results_list.append(results)

  # Sort results based on best validation data score
  sorted_results = sorted(results_list, key=itemgetter('validation_score'))


  return sorted_results

# Selects and trains the optimum Lasso model.
# Examines all defined features combos at poly powers 1,2 and 3
def train_lasso_model (feature_combos, X_train, y_train, X_test, y_test, cv, scoring, max_order):

  results = find_optimum_model(feature_combos, X_train, y_train, X_test, y_test,
              cv=cv, scoring=scoring, max_order=max_order)

  print("\n\n Lasso results:\n")
  for result in results:
    print("features: ", result["features"], ",  order: ", result["order"], ", validation score: ", np.sqrt(abs(result["validation_score"])), ",  test_score: ", np.sqrt(abs(result["test_score"])), ", alpha: ", result["alpha"])

  # Retrieve estimator with optimum score.
  return results[-1]