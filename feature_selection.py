#importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


"""
Code found on the internet is below
https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
"""
##############################################################################
#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
#############################################################################
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 7)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)
#############################################################################
#no of features
nof_list=np.arange(1,13)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
#############################################################################
cols = list(X.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 10)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
#############################################################################
"""
My attempt
"""
#def backward_selection(feat_mat, mfcc_mat, y_pred, y_test):
#    
#    threshold = None
#    featureMatrix = np.concatenate((feat_mat, mfcc_mat), axis=0)
#    featureValues = np.zeros(len(featureMatrix[0]))
#    featureDict = {"spectral centroid", "rms", "zero crossings", "crest", "flux",
#                   "MFCC1", "MFCC2", "MFCC3", "MFCC4", "MFCC5"}
#    
#    
#    for i in range(len(featureValues)):
#        featureMatrix = np.delete(featureMatrix, i, axis=0)
#        ##########################################
#        """This section is shaky because I'm not sure how to incorporate
#        the existing code into what I have. Below is my best guess."""
#        
#        X_train = scaler.transform(X_train)
#        X_test = scaler.transform(X_test)
#        
#        #svclassifier = SVC(kernel='linear')
#        #svclassifier = SVC(kernel='poly',degree=8)
#        #svclassifier = SVC(kernel='rbf')
#        svclassifier = SVC(kernel='sigmoid')
#        
#        svclassifier.fit(X_train, y_train)
#        newy_pred = svclassifier.predict(X_test)
#        featureValues = np.append(newy_pred)
#        ##########################################
#    
#    minValueIndex = np.argmax(featureValues)
#    featureValues = np.delete(featureValues, minValueIndex)
