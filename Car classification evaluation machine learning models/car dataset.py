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
decision_tree_model = 'finalized_model.pkl'
pickle.dump(d_tree, open(decision_tree_model, 'wb'))
