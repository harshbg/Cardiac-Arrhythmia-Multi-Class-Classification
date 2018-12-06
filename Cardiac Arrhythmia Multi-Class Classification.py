
# coding: utf-8

# ### Cardiac Arrhythmia Multy-Class Classification 
# 
# Analyze data and address missing data if there is any. 
# 
# Decide aboute a good evaluation strategy and justify your choice. 
# 
# Find the best parameters for the following classification models: 
# - KNN classifcation 
# - Logistic Regression
# - Linear Supprt Vector Machine
# - Kerenilzed Support Vector Machine
# - Decision Tree
# - Random Forest 

# # 1. Data Reading
# Starting with reading the data file and creating a dataframe for putting the output of all the models that 
#  we will be running. The purpose of doing this is it will be easy for us to compare the models.

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn import preprocessing
from matplotlib import pyplot as plt

data = pd.read_csv('cardiac_arrhythmia1.csv')
output = pd.DataFrame(index=None, columns=['model','train_Rsquare', 'test_Rsquare', 'train_MSE','test_MSE'])
data.describe()


# # Handling the missing value
# While going through the dataset we observed that out of 279 columns 5 columns have missing value in the form of '?'.
# The approach which we will following is, first replacing '?' with numpy NAN and then imputing the mean.

# In[4]:


import numpy as np
data['J'] = data['J'].replace('?',np.NaN)
data['Heart_Rate'] = data['Heart_Rate'].replace('?',np.NaN)
data['P'] = data['P'].replace('?',np.NaN)
data['T'] = data['T'].replace('?',np.NaN)
data['QRST'] = data['QRST'].replace('?',np.NaN)


# # Spliting the dataset
# Segregating the whole dataset into X and Y 

# In[5]:


Data_Y = data.cardiac_arrhythmia.values.ravel()
Data_X=data.drop('cardiac_arrhythmia', 1)


# In[6]:


np.unique(Data_Y, return_counts=True)


# # Drop column
# We can observe that column J have a lot of missing value. It will not be a good practice to impute mean values in this column.
# Better option will be that we drop this column

# In[7]:


Data_X.drop(columns=['J'])


# # Handling missing value
# We are imputing mean in place of missing values

# In[8]:


from sklearn.preprocessing import Imputer
z=Imputer(missing_values=np.nan, strategy='mean', axis=1).fit_transform(Data_X)
Data_X = pd.DataFrame(data=z,columns=Data_X.columns.values)
Data_X.isnull().sum()


# #  Good evaluation strategy
# As the dependent variabe is a categorical variable we will be using classification models. The best evaluation strategy for classification models is comparing the precision and recall. We know for a fact that R-sqaured and MSE scores are used extensively for checking the accuracy of a regression model where independent variable is a continous. However when we run classification models precision and recall are the best estimators of accuracy. Our main aim is to reduce the recall by improving the model.

# # Spliting down into both train and test data set

# In[9]:


data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(Data_X, Data_Y
                                                                        , random_state=2)
# Scaling the data (MIN MAX Scaling)
print('Shape of train {}, shape of test {}'.format(data_train_x.shape, data_test_x.shape))


# # Scaling
# As the variables are on different scale it will be helpful if we bring them all on the same scale. Scaling improves the performance of the models

# In[10]:


from sklearn.preprocessing import MinMaxScaler

#MinMax
MinMax = MinMaxScaler(feature_range= (0,1))
data_train_x = MinMax.fit_transform(data_train_x)
data_test_x = MinMax.transform(data_test_x)


# # 2. Modeling
# After taking care of the data we will be starting with the model creation. 

# # KNN Regression

# In[11]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

## We are creating a grid for which all n_neighbors values are to be used for cross validation

param_grid={'weights':['distance', 'uniform'], 'n_neighbors':range(1,100)}

## Using Grid search for exhaustive searching

grid_search = GridSearchCV( KNeighborsClassifier(),param_grid, cv = 10)
grid_search.fit(data_train_x, data_train_y)


# In[12]:


from sklearn.metrics import r2_score, mean_squared_error
train_Rsquare = grid_search.score(data_train_x, data_train_y)
test_Rsquare = grid_search.score(data_test_x, data_test_y)
train_MSE = mean_squared_error(data_train_y, grid_search.predict(data_train_x))
test_MSE = mean_squared_error(data_test_y, grid_search.predict(data_test_x))
output = output.append(pd.Series({'model':'KNN Classifier','train_Rsquare':train_Rsquare, 'test_Rsquare':test_Rsquare, 'train_MSE':train_MSE,'test_MSE':test_MSE}),ignore_index=True )
output


# # R-squared, MSE vs precision, Recall
# We know for a fact that R-sqaured and MSE scores are used extensively for checking the accuracy of a regression model where independent variable is a continous. However when we run classification models precision and recall are the best estimators of accuracy

# In[13]:


pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_estimator_)


# In[14]:


from sklearn.metrics import classification_report
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=6, p=2,
           weights='distance')
knn.fit(data_train_x, data_train_y)
pred = knn.predict(data_test_x)
print(classification_report(data_test_y,pred))


# ## Output Evaluation
# In the whole process of model creation, our aim will be to achieve maximum Precision (no false positive) and maximum Recall (no false negative) there needs to be an absence of type I and II errors

# # Logistic Regression

# In[15]:


from sklearn.linear_model import LogisticRegression

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

grid_search_log = GridSearchCV(LogisticRegression(penalty='l2'), param_grid, cv=5)
grid_search_log.fit(data_train_x, data_train_y)


# In[16]:


from sklearn.metrics import r2_score, mean_squared_error
train_Rsquare = grid_search_log.score(data_train_x, data_train_y)
test_Rsquare = grid_search_log.score(data_test_x, data_test_y)
train_MSE = mean_squared_error(data_train_y, grid_search_log.predict(data_train_x))
test_MSE = mean_squared_error(data_test_y, grid_search_log.predict(data_test_x))
output = output.append(pd.Series({'model':'Logistic Regression','train_Rsquare':train_Rsquare, 'test_Rsquare':test_Rsquare, 'train_MSE':train_MSE,'test_MSE':test_MSE}),ignore_index=True )
output


# In[17]:


pd.DataFrame(grid_search_log.cv_results_)
print(grid_search_log.best_estimator_)


# In[18]:


log = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
log.fit(data_train_x, data_train_y)
pred = log.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Linear Supprt Vector Machine

# In[19]:


from sklearn.svm import LinearSVC

param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100, 1000], 'max_iter':[1000,10000] }

grid_search_SVC = GridSearchCV(LinearSVC(random_state=0), param_grid, cv=5)
grid_search_SVC.fit(data_train_x, data_train_y)


# In[20]:


train_Rsquare = grid_search_SVC.score(data_train_x, data_train_y)
test_Rsquare = grid_search_SVC.score(data_test_x, data_test_y)
train_MSE = mean_squared_error(data_train_y, grid_search_SVC.predict(data_train_x))
test_MSE = mean_squared_error(data_test_y, grid_search_SVC.predict(data_test_x))
output = output.append(pd.Series({'model':'Linear SVC','train_Rsquare':train_Rsquare, 'test_Rsquare':test_Rsquare, 'train_MSE':train_MSE,'test_MSE':test_MSE}),ignore_index=True )
output


# In[21]:


pd.DataFrame(grid_search_SVC.cv_results_)
print(grid_search_SVC.best_estimator_)


# In[22]:


linearsvc = LinearSVC(C=0.1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
linearsvc.fit(data_train_x, data_train_y)
pred = linearsvc.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Kerenilzed Support Vector Machine

# In[23]:


from sklearn.svm import SVC

param_grid = {'C':[0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100, 1000], 'gamma':[0.001, 0.01, 0.1, 0.5, 1, 10]}

grid_search_KSVC = GridSearchCV(SVC(kernel = 'rbf'), param_grid, cv=5)
grid_search_KSVC.fit(data_train_x, data_train_y)


# In[24]:


train_Rsquare = grid_search_KSVC.score(data_train_x, data_train_y)
test_Rsquare = grid_search_KSVC.score(data_test_x, data_test_y)
train_MSE = mean_squared_error(data_train_y, grid_search_KSVC.predict(data_train_x))
test_MSE = mean_squared_error(data_test_y, grid_search_KSVC.predict(data_test_x))
output = output.append(pd.Series({'model':'Kernel SVC','train_Rsquare':train_Rsquare, 'test_Rsquare':test_Rsquare, 'train_MSE':train_MSE,'test_MSE':test_MSE}),ignore_index=True )
output


# In[25]:


pd.DataFrame(grid_search_KSVC.cv_results_)
print(grid_search_KSVC.best_estimator_)


# In[26]:


svc = SVC(C=50, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
svc.fit(data_train_x, data_train_y)
pred = svc.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Decision Tree

# In[27]:


from sklearn.tree import DecisionTreeClassifier

param_grid = {'max_features':[None,'auto', 'log2'], 'max_depth':[5,10,15,20,50]}

grid_search_DT = GridSearchCV(DecisionTreeClassifier(random_state = 10), param_grid, cv=5)
grid_search_DT.fit(data_train_x, data_train_y)


# In[28]:


train_Rsquare = grid_search_DT.score(data_train_x, data_train_y)
test_Rsquare = grid_search_DT.score(data_test_x, data_test_y)
train_MSE = mean_squared_error(data_train_y, grid_search_DT.predict(data_train_x))
test_MSE = mean_squared_error(data_test_y, grid_search_DT.predict(data_test_x))
output = output.append(pd.Series({'model':'Decision Tree Classifier','train_Rsquare':train_Rsquare, 'test_Rsquare':test_Rsquare, 'train_MSE':train_MSE,'test_MSE':test_MSE}),ignore_index=True )
output


# In[29]:


pd.DataFrame(grid_search_DT.cv_results_)
print(grid_search_DT.best_estimator_)


# In[30]:


dt = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=10,
            splitter='best')
dt.fit(data_train_x, data_train_y)
pred = dt.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Random Forest

# In[31]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

#Tuning ridge on new dataset
param_grid = {"max_depth": [3, 5],
              "max_features": sp_randint(1, 40),
              "min_samples_split": sp_randint(2, 30),
              "min_samples_leaf": sp_randint(1, 20),
              "bootstrap": [True, False]}
grid_search_RF = RandomizedSearchCV(RandomForestClassifier(n_estimators=1000), param_distributions=param_grid,
                                   n_iter=30, random_state=0,n_jobs=-1)
grid_search_RF.fit(data_train_x, data_train_y)


# In[32]:


train_Rsquare = grid_search_RF.score(data_train_x, data_train_y)
test_Rsquare = grid_search_RF.score(data_test_x, data_test_y)
train_MSE = mean_squared_error(data_train_y, grid_search_RF.predict(data_train_x))
test_MSE = mean_squared_error(data_test_y, grid_search_RF.predict(data_test_x))
output = output.append(pd.Series({'model':'Random Forest Classifier','train_Rsquare':train_Rsquare, 'test_Rsquare':test_Rsquare, 'train_MSE':train_MSE,'test_MSE':test_MSE}),ignore_index=True )
output


# In[33]:


pd.DataFrame(grid_search_RF.cv_results_)
print(grid_search_RF.best_estimator_)


# In[34]:


rf = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=5, max_features=36, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=7, min_samples_split=25,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
rf.fit(data_train_x, data_train_y)
pred = rf.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # 3. Bagging
# We will be using bagging to improve our earlier model with best parameters

# ##   KNN with bagging

# In[35]:


from sklearn.ensemble import BaggingClassifier

KNN_bagging = BaggingClassifier(knn, n_estimators = 100, bootstrap = True)
KNN_bagging.fit(data_train_x,data_train_y)
pred = KNN_bagging.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Logistic regression with bagging

# In[36]:


log_bagging = BaggingClassifier(log , n_estimators = 100, max_features = 200 ,bootstrap = True, oob_score = True)
log_bagging.fit(data_train_x, data_train_y)
pred = log_bagging.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Linear SVC with bagging

# In[37]:


linearsvc_bagging = BaggingClassifier(linearsvc , n_estimators = 100, max_features = 200 ,
                                      bootstrap = True, oob_score = True)
linearsvc_bagging.fit(data_train_x, data_train_y)
pred = linearsvc_bagging.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Kernalized SVC with bagging

# In[38]:


svc_bagging = BaggingClassifier(svc , n_estimators = 100, max_features = 200 ,
                                      bootstrap = True, oob_score = True)
svc_bagging.fit(data_train_x, data_train_y)
pred = svc_bagging.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Decision Tree with bagging

# In[39]:


dt_bagging = BaggingClassifier(dt , n_estimators = 100, max_features = 200 ,
                                      bootstrap = True, oob_score = True)
dt_bagging.fit(data_train_x, data_train_y)
pred = dt_bagging.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Random forest with bagging

# In[40]:


rf_bagging = BaggingClassifier(rf , n_estimators = 100, max_features = 200 ,
                                      bootstrap = True, oob_score = True)
rf_bagging.fit(data_train_x, data_train_y)
pred = rf_bagging.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Boosting
# The main idea behind using boosting is to convert weak learners 

# # Ada boosting with Logistic regression

# In[41]:


from sklearn.ensemble import AdaBoostClassifier
param_grid = {'learning_rate':[0.0001,0.001,0.01,0.1,1]}
adaboost_log = GridSearchCV(AdaBoostClassifier(base_estimator = log,random_state = 0), param_grid, cv=5,return_train_score=True)
adaboost_log.fit(data_train_x, data_train_y)

pred = adaboost_log.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Ada boosting with Linear SVM

# In[42]:


from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier
param_grid = {'learning_rate':[0.0001,0.001,0.01,0.1,1]}
adaboost_svc = GridSearchCV(AdaBoostClassifier(base_estimator = linearsvc,random_state = 0, algorithm='SAMME'),
                            param_grid, cv=5,return_train_score=True)
adaboost_svc.fit(data_train_x, data_train_y)

pred = adaboost_svc.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Ada boosting with Kernalized SVM

# In[43]:


from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier
param_grid = {'learning_rate':[0.0001,0.001,0.01,0.1,1]}
adaboost_ksvc = GridSearchCV(AdaBoostClassifier(base_estimator = svc,random_state = 0, algorithm='SAMME'),
                            param_grid, cv=5,return_train_score=True)
adaboost_ksvc.fit(data_train_x, data_train_y)

pred = adaboost_ksvc.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Ada Boosting with Decision Tree

# In[44]:


from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier
param_grid = {'learning_rate':[0.0001,0.001,0.01,0.1,1]}
adaboost_dt = GridSearchCV(AdaBoostClassifier(base_estimator = dt,random_state = 0),
                            param_grid, cv=5,return_train_score=True)
adaboost_dt.fit(data_train_x, data_train_y)

pred = adaboost_dt.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Ada Boosting with Random Forest

# # Gradient Boosting Classifier

# In[46]:


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(random_state=10, n_estimators= 500)

param_grid = {'max_features':['auto', 'log2'], 'learning_rate' : [0.01,0.1], 'max_depth':[5,10,15,30,50]}

grid_search_gb = GridSearchCV(model, param_grid, cv=5)
grid_search_gb.fit(data_train_x, data_train_y)
pred = grid_search_gb.predict(data_test_x)
print(classification_report(data_test_y,pred))


# # Comparing bagging and boosting models with previous one's
# When we compare the precision and recall of earlier model with models with baaging and boosting we see that the new models accuracy has dropped. Bagging and boosting are used to lower the bias of the model, When we apply boost we introduce the risk of overfitting. Overfitting ulitmately lowers the accuracy estimated by cross-validation

# # 4. PCA
# We will be using principal component analysis for data reduction. We have 278 variables which is increasing the complexity of the models. PCA reduces the complexity of the model by only considering the important variables and thus reducing the overall data set.

# In[49]:


from sklearn.decomposition import PCA

pca = PCA(0.95)
pca.fit(Data_X)
Data_X_PCA = pca.transform(Data_X)

data_train_x_pca, data_test_x_pca, data_train_y_pca, data_test_y_pca = train_test_split(Data_X_PCA, Data_Y, random_state = 100)


# In[50]:


#Checking results of 2 scalars

from sklearn.preprocessing import MinMaxScaler

#MinMax
MinMax = MinMaxScaler(feature_range= (0,1))
data_train_x_pca = MinMax.fit_transform(data_train_x_pca)
data_test_x_pca = MinMax.transform(data_test_x_pca)


# # KNN with PCA

# In[51]:


from sklearn.neighbors import KNeighborsClassifier

param_grid = {'weights':['distance', 'uniform'], 'n_neighbors':range(3,100)}

grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5,return_train_score=True)
grid_search_knn.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(grid_search_knn.cv_results_)
print(grid_search_knn.best_score_)


# In[52]:


print(grid_search_knn.best_estimator_)


# In[53]:


knn_pca = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='distance')
knn_pca.fit(data_train_x_pca, data_train_y_pca)
pred = knn_pca.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))


# # Logistic with PCA

# In[54]:


from sklearn.linear_model import LogisticRegression

param_grid = {'C': range(1,100) }

grid_search_log = GridSearchCV(LogisticRegression(penalty='l1'), param_grid, cv=5,return_train_score=True)
grid_search_log.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(grid_search_log.cv_results_)
print(grid_search_log.best_estimator_)


# In[55]:


print(grid_search_log.best_estimator_)


# In[56]:


logistic_pca = LogisticRegression(C=4, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
logistic_pca.fit(data_train_x_pca, data_train_y_pca)
pred = logistic_pca.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))


# # Linear SVM with PCA

# In[58]:


from sklearn.svm import LinearSVC

param_grid = {'C': [0.0001,0.001,0.01,0.1,2,3,4,5,6,7,8,9], 'max_iter':[100,1000,10000] }

grid_search_linearsvc = GridSearchCV(LinearSVC(random_state=0), param_grid, cv=5,return_train_score=True)
grid_search_linearsvc.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(grid_search_linearsvc.cv_results_)
print(grid_search_linearsvc.best_estimator_)


# In[59]:


linearsvc = LinearSVC(C=2, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=100,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
linearsvc.fit(data_train_x_pca, data_train_y_pca)
pred = linearsvc.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))


# # Kernalized SVM with PCA

# In[60]:


from sklearn.svm import SVC

param_grid = {'C':[0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100, 1000], 'gamma':[0.001, 0.01, 0.1, 0.5, 1, 10]}

grid_search_svc = GridSearchCV(SVC(kernel = 'rbf'), param_grid, cv=5,return_train_score=True)
grid_search_svc.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(grid_search_svc.cv_results_)
print(grid_search_svc.best_estimator_)


# In[61]:


svc = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
svc.fit(data_train_x_pca, data_train_y_pca)
pred = svc.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))


# # Decision tree with PCA

# In[62]:


from sklearn.tree import DecisionTreeClassifier

param_grid = {'max_features':['auto', 'log2'], 'max_depth':[3,5,7,9,10,11,13,15,20,50]}

grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state = 0), param_grid, cv=5,return_train_score=True)
grid_search_dt.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(grid_search_dt.cv_results_)
print(grid_search_dt.best_estimator_)


# In[63]:


dt_pca = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
            splitter='best')
dt_pca.fit(data_train_x_pca, data_train_y_pca)
pred = dt_pca.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))


# # Random Forest with PCA

# In[64]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

#Tuning ridge on new dataset
param_grid = {"max_depth": sp_randint(3, 9),
              "max_features": sp_randint(1, 10),
              "min_samples_split": sp_randint(2, 20),
              "min_samples_leaf": sp_randint(1, 20),
              "bootstrap": [True, False]}
random_search_rf = RandomizedSearchCV(RandomForestClassifier(n_estimators=1000), param_distributions=param_grid,
                                   n_iter=30, random_state=0,n_jobs=-1, cv= 5,return_train_score=True)
random_search_rf.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(random_search_rf.cv_results_)
print(random_search_rf.best_estimator_)


# In[65]:


rf_pca = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=5, max_features=9, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=3,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
rf_pca.fit(data_train_x_pca, data_train_y_pca)
pred = rf_pca.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))


# # Gradient boosting with PCA

# In[66]:


param_grid = {"max_depth": sp_randint(5,30),
              "max_features": sp_randint(1, 35),
              "min_samples_split": sp_randint(2, 40),
              "min_samples_leaf": sp_randint(1, 30),
             'learning_rate': [0.01,0.001,0.1]}
random_search_gb = RandomizedSearchCV(GradientBoostingClassifier(n_estimators=1000), param_distributions=param_grid,
                                   n_iter=30, random_state=0,n_jobs=-1,cv= 5,return_train_score=True)
random_search_gb.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(random_search_gb.cv_results_)
print(random_search_gb.best_estimator_)


# In[67]:


gbc_pca = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.01, loss='deviance', max_depth=9,
              max_features=6, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=7, min_samples_split=19,
              min_weight_fraction_leaf=0.0, n_estimators=1000,
              presort='auto', random_state=None, subsample=1.0, verbose=0,
              warm_start=False)
gbc_pca.fit(data_train_x_pca, data_train_y_pca)
pred = gbc_pca.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))


# # ADA boosting with PCA

# In[68]:


param_grid = {'learning_rate' :[0.0001,0.001,0.01,0.1,1,1.2,2] ,
              "algorithm":['SAMME', 'SAMME.R']}
random_search_ada = RandomizedSearchCV(AdaBoostClassifier(n_estimators = 1000), param_distributions=param_grid,
                                   n_iter=10, random_state=0,n_jobs=-1,cv= 5,return_train_score=True)
random_search_ada.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(random_search_ada.cv_results_)
print(random_search_ada.best_estimator_)


# In[69]:


ada_pca = AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=1,
          n_estimators=1000, random_state=None)
ada_pca.fit(data_train_x_pca, data_train_y_pca)
pred = ada_pca.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))


# # Evaluation after PCA
# The models started performing better after we applied PCA on the original data. The reason behind this is, PCA reduces the complexity of the data. It creates components based on giving importance to variables with large variance and also the components which it creates are non collinear in nature which means it takes care of collinearity in large data set. PCA also improves the overall execution time and quality of the models and it is very beneficial when we are working with huge amount of variables.
# The Best model according to the precision and recall score is Kernalized SVM with PCA having accuracy of 75%.

# ## End
