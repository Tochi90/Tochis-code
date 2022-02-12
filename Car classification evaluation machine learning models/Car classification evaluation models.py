#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import shap
import pickle
import eli5
from eli5 import show_weights
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,GridSearchCV,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn import metrics


# In[2]:


cars = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
                  names = ["Buying", "Maintenance", "Doors", "Person", "Luggage boot", "Safety", "Class"])


# In[3]:


#view the first five rows
cars.head()


# In[4]:


#checking for missing values
print (cars.isnull().sum())


# In[5]:


#choosing dependent variable
Y = cars.Class
Y


# In[6]:


#subsetting the dependent variable to get the independent variables
X = cars.drop(['Class'], axis=1)
X


# In[7]:


#changing strings to float so it can be fit into the model without throwing errors
X.iloc[:,0:2].replace({'low':0,'med':1/3,'high':2/3, 'vhigh':1}, inplace = True)
X.iloc[:,2].replace({'2':0,'3':1/3,'4':2/3,'5more':1},inplace = True)
X.iloc[:,3].replace({'2':0,'4':0.5,'more':1},inplace = True)
X.iloc[:,4].replace({'small':0,'med':0.5,'big':1},inplace = True)
X.iloc[:,5].replace({'low':0,'med':0.5,'high':1},inplace = True)


# In[8]:


#show the shape of the independent variables
X.shape


# In[9]:


#save the independent variables in the col_features
col_features = X.columns.values
col_features


# In[10]:


#show the shape of the target column
Y.shape


# In[11]:


#importing the train-test split to be used for both models
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=45)
f_measure_score = {'decision_tree':{},'svm':{},'knn':{},'logistic':{}}


# In[12]:


#setting aside the cross validation data
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=45)


# In[13]:


#building the decision tree
parameter_tree = {'criterion':['entropy'],'max_depth':list(range(8,11)),'min_samples_leaf':list(range(1,4)),'random_state':[45]}
d_tree = DecisionTreeClassifier()
grid_tree = GridSearchCV(d_tree, parameter_tree, cv = cv, scoring='f1_micro')
grid_tree.fit(X_train,Y_train)
y_pred_tree = grid_tree.predict(X_test)
nested_score_tree = cross_val_score(grid_tree, X = X, y = Y, cv = cv) 
f_measure_score['decision_tree']['mean'] = np.mean(nested_score_tree)
f_measure_score['decision_tree']['std'] = np.std(nested_score_tree)


# In[14]:


#print values of the classification from the decision tree model
print('precision,recall,f-measure\n', classification_report(Y_test,y_pred_tree))


# In[15]:


#Modelling the pipeline for the decision tree
X, Y = make_classification(random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                     random_state=45)
pipe = Pipeline([('scaler', StandardScaler()), ('d_tree', DecisionTreeClassifier())])

# The pipeline can be used as any other estimator
# and this avoids leaking the test set into the train set
pipe.fit(X_train, Y_train)
Pipeline(steps=[('scaler', StandardScaler()), ('d_tree', DecisionTreeClassifier())])
pipe.score(X_test, Y_test)


# In[16]:


#best parameters of the decision tree
grid_tree.best_params_


# In[17]:


#generating the ROC CURVE
parameter_tree = {'criterion':['entropy'],'max_depth':list(range(8,11)),'min_samples_leaf':list(range(1,4)),'random_state':[45]}
d_tree = DecisionTreeClassifier()
grid_tree = GridSearchCV(d_tree, parameter_tree, cv = cv, scoring='f1_micro')
d_tree.fit(X_train,Y_train)

# predict probabilities
pred_prob1 = d_tree.predict_proba(X_test)

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(Y_test, pred_prob1[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(Y_test))]
p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=1)

# auc scores
auc_score1 = roc_auc_score(Y_test, pred_prob1[:,1])

print(auc_score1)


# In[18]:


#Plotting the ROC curve
plt.plot(fpr1,tpr1,label="AUC="+str(auc_score1))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# In[19]:


#The AUC for this decision tree model turns out to be 0.9. 
#Since this is closest to 1.0, this confirms that the model does the BEST job of classifying data
#because the closer AUC is to 1, the better the model. 
#A model with an AUC equal to 0.5 is no better than a model that makes random classifications.


# In[20]:


#generating feature importance
dtree_importance = d_tree.feature_importances_
# summarize feature importance
for i,v in enumerate(dtree_importance):
	print('Feature: %0d, Score: %.5f' % (i,v))


# In[21]:


#using shap to explain feature importance

dtree_explainer = shap.TreeExplainer(d_tree)
shap_values = dtree_explainer.shap_values(X)


# In[22]:


shap_values
shap.summary_plot(shap_values, X, plot_type='bar')


# In[23]:


#using pickle to save model so it can be used for future purposes

# save the model to disk
decision_tree_model = 'finalized_model.sav'
pickle.dump(d_tree, open(decision_tree_model, 'wb'))


# In[24]:


###LOGISTIC REGRESSION MODEL

#building the logistic regression model I used (100 and 1000 iterations but it could not complete so i stayed with 10 iterations)
parameter_log = {'C':[10]}
logistic = LogisticRegression(multi_class='multinomial',solver='lbfgs',penalty = 'l2',random_state = 45)
grid_log = GridSearchCV(logistic, parameter_log, cv = cv, scoring='f1_micro')
grid_log.fit(X_train, Y_train)
y_pred_log = grid_log.predict(X_test)
nested_score_log = cross_val_score(grid_log, X = X, y = Y, cv = cv) 
f_measure_score['logistic']['mean'] = np.mean(nested_score_log)
f_measure_score['logistic']['std'] = np.std(nested_score_log)


# In[25]:


#print values of the classification from the logisitc regression model
print('precision,recall,f-measure\n', classification_report(Y_test,y_pred_log),'\n')


# In[26]:


#Modelling the pipeline

X, Y = make_classification(random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                     random_state=45)
pipe = Pipeline([('scaler', StandardScaler()), ('logisitic', LogisticRegression())])
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
pipe.fit(X_train, Y_train)
Pipeline(steps=[('scaler', StandardScaler()), ('logisitic', LogisticRegression())])
pipe.score(X_test, Y_test)


# In[27]:


#best parameters of the logistic regression model
grid_log.best_params_


# In[28]:


#creating the ROC CURVE for the logistic regression model
parameter_log = {'C':[10]}
logistic = LogisticRegression(multi_class='multinomial',solver='lbfgs',penalty = 'l2',random_state = 45)
grid_log = GridSearchCV(logistic, parameter_log, cv = cv, scoring='f1_micro')
logistic.fit(X_train, Y_train)

# predict probabilities
pred_prob2 = logistic.predict_proba(X_test)

# roc curve for models
fpr2, tpr2, thresh2 = roc_curve(Y_test, pred_prob2[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(Y_test))]
p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=1)

# auc scores
auc_score2 = roc_auc_score(Y_test, pred_prob2[:,1])

print(auc_score2)


# In[29]:


#plotting the ROC curve and calculating the Area under the curve
plt.plot(fpr2,tpr2,label="AUC="+str(auc_score2))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# In[30]:


#The AUC for this logistic regression model turns out to be 0.82. 
#Since this is close to 1.0, this confirms that the model does a GOOD job of classifying data.


# In[31]:


# getting the feature importance
logistic_importance = logistic.coef_[0]
# summarize feature importance
for i,v in enumerate(logistic_importance):
	print('Feature: %0d, Score: %.5f' % (i,v))


# In[32]:


#using eli5 to explain feature importance

eli5.explain_weights(logistic)


# In[33]:


#using pickle to save model so it can be used for future purposes

# save the model to disk
logistic_regression_model = 'finalized_model.sav'
pickle.dump(logistic, open(logistic_regression_model, 'wb'))


# In[34]:


###K-NEAREST NEIGHBOR MODEL

#Building the Knn model

para_knn = {'n_neighbors':list(range(12,17)),'weights':['uniform','distance']}
knn = KNeighborsClassifier()
grid_knn = GridSearchCV(knn, para_knn, cv = cv, scoring='f1_micro')
grid_knn.fit(X_train,Y_train)
y_pred_knn = grid_knn.predict(X_test)
nested_score_knn = cross_val_score(grid_knn, X = X, y = Y, cv = cv) 
f_measure_score['knn']['mean'] = np.mean(nested_score_knn)
f_measure_score['knn']['std'] = np.std(nested_score_knn)


# In[35]:


#Printing the classification of the knn model

print('precision,recall,f-measure\n', classification_report(Y_test,y_pred_knn))


# In[36]:


#Modelling the Pipeline

X, Y = make_classification(random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                     random_state=45)
pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
pipe.fit(X_train, Y_train)
Pipeline(steps=[('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
pipe.score(X_test, Y_test)


# In[37]:


#best parameter of the KNN model
grid_knn.best_params_


# In[38]:


#Generating the ROC curve and calculating the Area under the curve

para_knn = {'n_neighbors':list(range(12,17)),'weights':['uniform','distance']}
knn = KNeighborsClassifier()
grid_knn = GridSearchCV(knn, para_knn, cv = cv, scoring='f1_micro')
knn.fit(X_train,Y_train)

# predict probabilities
pred_prob3 = knn.predict_proba(X_test)

# roc curve for models
fpr3, tpr3, thresh3 = roc_curve(Y_test, pred_prob3[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(Y_test))]
p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=1)

# auc scores
auc_score3 = roc_auc_score(Y_test, pred_prob3[:,1])

print(auc_score3)


# In[39]:


#plotting the ROC curve
plt.plot(fpr3,tpr3,label="AUC="+str(auc_score3))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# In[40]:


#The AUC for this knn model turns out to be 0.83. 
#Since this is close to 1.0, this confirms that the model does a GOOD job of classifying data.


# In[41]:


# perform permutation importance to generate the feature importance of the model
results = permutation_importance(knn, X, Y, scoring='accuracy')
# get importance
knn_importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(knn_importance):
	print('Feature: %0d, Score: %.5f' % (i,v))


# In[42]:


#using shap to explain the feature importance

# Produce the SHAP values
knn_explainer = shap.KernelExplainer(knn.predict,X_test)
knn_shap_values = knn_explainer.shap_values(X_test)


# In[43]:


#plotting the SHAP values

shap.summary_plot(knn_shap_values, X_test)


# In[44]:


#using pickle to save model so it can be used for future purposes

# save the model to disk
knn_model = 'finalized_model.sav'
pickle.dump(knn, open(knn_model, 'wb'))


# In[45]:


### SVM MODEL

#Building the svm model

para_svm = {'kernel':['rbf'],'C':[10],'gamma':[5]} 
svm = SVC(random_state = 45,probability = True)
grid_svm = GridSearchCV(svm, para_svm, cv = cv, scoring='f1_micro')
grid_svm.fit(X_train, Y_train)
y_pred_svm = grid_svm.predict(X_test)
nested_score_svm = cross_val_score(grid_svm, X = X, y = Y, cv = cv) 
f_measure_score['svm']['mean'] = np.mean(nested_score_svm)
f_measure_score['svm']['std'] = np.std(nested_score_svm)


# In[46]:


#printing the classification of the SVM model
print('precision,recall,f-measure\n', classification_report(Y_test,y_pred_svm),'\n')


# In[47]:


#Modeling the pipeline

X, Y = make_classification(random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                     random_state=45)
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
pipe.fit(X_train, Y_train)
Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC())])
pipe.score(X_test, Y_test)


# In[48]:


#best parameters of the model

grid_svm.best_params_


# In[49]:


#Creating the ROC and calculating the Area under the curve

para_svm = {'kernel':['rbf'],'C':[10],'gamma':[5]} 
svm = SVC(random_state = 45,probability = True)
grid_svm = GridSearchCV(svm, para_svm, cv = cv, scoring='f1_micro')
svm.fit(X_train, Y_train)

# predict probabilities
pred_prob4 = svm.predict_proba(X_test)

# roc curve for models
fpr4, tpr4, thresh4 = roc_curve(Y_test, pred_prob4[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(Y_test))]
p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=1)

# auc scores
auc_score4 = roc_auc_score(Y_test, pred_prob4[:,1])

print(auc_score4)


# In[50]:


#Plotting the ROC curve
plt.plot(fpr4,tpr4,label="AUC="+str(auc_score4))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# In[51]:


#The AUC for this svm model turns out to be 0.89. 
#Since this is closer to 1.0, this confirms that the model does a BETTER job of classifying data.


# In[52]:


#get feature importance for the model
# The SHAP values
svm_explainer = shap.KernelExplainer(svm.predict,X_test)
svm_shap_values = svm_explainer.shap_values(X_test)


# In[53]:


#plotting the SHAP values

shap.summary_plot(svm_shap_values, X_test)


# In[54]:


#using pickle to save model so it can be used for future purposes

# save the model to disk
svm_model = 'finalized_model.sav'
pickle.dump(svm, open(svm_model, 'wb'))


# In[55]:


#comparing the 4 models designed
for k,v in f_measure_score.items():
    print(k, ': ', v)


# In[56]:


#the model with the best accuracy is the decision tree model
accuracy_tree = d_tree.score(X_test,Y_test)
print('accuracy of decisiontree: ', accuracy_tree)


# In[ ]:




