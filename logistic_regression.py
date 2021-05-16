#logistic regression 11/21/19
#purpose: run logistic regression for each instrument class, identify features that are associated with at least one instrument

import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

dirname='C:/Users/Laney/Documents/6201_comp_mus_analysis/Final project/data/'
filename='allDataNorm.csv'
datadf = pd.read_csv(dirname+filename)

#if listwise deletion of rows with missing feature values (these are onsets not identified by our onset detection)
datadf_nomiss=datadf.dropna(axis=0)
for col in datadf.columns:
    print(col)

X = datadf_nomiss.drop(['Unnamed: 0','index','label','filename','dirname_x'],axis=1)

#create dummy variables for each instrument from labels
y_dummies = pd.get_dummies(datadf_nomiss['label'])

instruments=['SD','HH','KD','CY']

featureList=['centroid','rms','zeroCrossings','crest','flux','mfcc1','mfcc2','mfcc3','mfcc4',
'mfcc5','flatness','rolloff','c0','c100','c200','c400','c800','c1600','c3200','RMSF','lowRMS','bandRMS',
'highRMS','lowRMSrel','bandRMSrel','highRMSrel','lowRMSrelband','lowRMSrelhigh','bandRMSrelhigh']

for inst in instruments:
    y = y_dummies[inst]
    logitreg=sm.Logit(y, X).fit(method='newton')
    print('y=',inst)
    print(logitreg.summary())


