# Cardiac Arrhythmia Multi Class Classification
> The study aims to correctly classify 15 different types of cardiac arrhythmia. The project was done as part of Machine Learning class at the University of Texas at Dallas.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Process](#process)
* [Code Examples](#code-examples)
* [Features](#features)
* [Status](#status)
* [Contact](#contact)

## General info
This study is conducted to classify patients into one of the sixteen subclasses, 
among which one class represents absence of disease and the other fifteen classes represent electrocardiogram records of various subtypes of arrhythmias.


## Technologies
* Python - version 3.5
* Tensorflow

## Setup

The dataset used and its metadata can be found [here](). The jupyter notebook can be downloaded [here]() and can be used to reproduce the result. Installation of TensorFlow would be required to run all the models. 
You can find the instructions to install [here]().

## Process

* As it is a huge dataset with nearly 280 variables first I performed feature selection technique to identify the important variables impacting the prediction. 
* I used various machine learning models like KNN, logistic regression, random forest, decision tree, linear & kernalised SVM and compared the precision and recall of the mentioned models. 
* To improve the accuracy of the models I used bagging and boosting and evaluated the performance of these models.
* As the data file had 280 variables, I used PCA to improve accuracy.


## Code Examples

`
# KNN Regression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

## We are creating a grid for which all n_neighbors values are to be used for cross validation

param_grid={'weights':['distance', 'uniform'], 'n_neighbors':range(1,100)}

## Using Grid search for exhaustive searching

grid_search = GridSearchCV( KNeighborsClassifier(),param_grid, cv = 10)
grid_search.fit(data_train_x, data_train_y)

`

`
# KNN with Bagging

from sklearn.ensemble import BaggingClassifier

KNN_bagging = BaggingClassifier(knn, n_estimators = 100, bootstrap = True)
KNN_bagging.fit(data_train_x,data_train_y)
pred = KNN_bagging.predict(data_test_x)
print(classification_report(data_test_y,pred))

`

`
# Ada boosting with Logistic regression

from sklearn.ensemble import AdaBoostClassifier
param_grid = {'learning_rate':[0.0001,0.001,0.01,0.1,1]}
adaboost_log = GridSearchCV(AdaBoostClassifier(base_estimator = log,random_state = 0), param_grid, cv=5,return_train_score=True)
adaboost_log.fit(data_train_x, data_train_y)

pred = adaboost_log.predict(data_test_x)
print(classification_report(data_test_y,pred))
`

`
# KNN with PCA

from sklearn.neighbors import KNeighborsClassifier

param_grid = {'weights':['distance', 'uniform'], 'n_neighbors':range(3,100)}

grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5,return_train_score=True)
grid_search_knn.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(grid_search_knn.cv_results_)
print(grid_search_knn.best_score_)
`

## Features
* We were able to improve the classification accuracy, and classify the type of cardiac arrhythmia with 75% accuracy. 

## Status
Project is:  _finished_

## Contact

Created by me and my teammate [Siddharth Oza]().
Feel free to contact me! My other projects can be found [here](http://www.gupta-harsh.com/projects/)