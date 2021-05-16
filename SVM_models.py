#baseline models.py
#purpose: 1) run baseline SVM (10 features) on MDB Drums data, testing different kernel shapes, 75/25 split
#2) run baseline model with 10-fold cross validation
#3) run baseline model with combined MDB and ENST datasets
#4) alternate feature set with combined MDB and ENST datasets, ENST only, and MDB only

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import random

dirname='C:/Users/Laney/Documents/6201_comp_mus_analysis/Final project/data/'
filenameAll='allDataNorm12_4_19.csv'
filenameMDB='MDBdataNorm12_4_19.csv'
filenameENST='ENSTdataNorm12_4_19.csv'

dfAll = pd.read_csv(dirname+filenameAll)
dfMDB = pd.read_csv(dirname+filenameMDB)
dfENST = pd.read_csv(dirname+filenameENST)

dfAll_nomiss=dfAll.dropna(axis=0)
dfMDB_nomiss=dfMDB.dropna(axis=0)
dfENST_nomiss=dfENST.dropna(axis=0)

##MDB data only, 75/25 train/test split
fileListMDB=dfMDB_nomiss.filename.unique()
#print ('MDB file list in original order=',fileListMDB)

#sort file list randomly, initialized with a seed to get the same result every time
random.seed(123)
random.shuffle(fileListMDB)

#assign 75% of MDB filenames to training, 25% to testing
isplit=round(len(fileListMDB)*0.75)-1
filemapMDB=[]

for i in range(0,len(fileListMDB)):
    if i<=isplit:
        filemapMDB.append([fileListMDB[i],'training'])
    else:
        filemapMDB.append([fileListMDB[i],'testing'])

fileMapDF=pd.DataFrame(filemapMDB,columns=['filename','testOrTrain'])

#check frequencies
fileMapDF.testOrTrain.value_counts()

#merge in the training/testing column and split data

testTrain = dfMDB_nomiss.merge(fileMapDF,how='inner',left_on=['filename'],right_on=['filename'])
test=testTrain[testTrain['testOrTrain'] == 'testing']
train=testTrain[testTrain['testOrTrain'] == 'training']

X_test= test.drop(['Unnamed: 0','label','filename','dirname_x','testOrTrain'],axis=1)
X_train=train.drop(['Unnamed: 0','label','filename','dirname_x','testOrTrain'],axis=1)
y_test= pd.DataFrame(test['label'])
y_train=pd.DataFrame(train['label'])

#MDB data with same test/train split but only 3 instrument classes
test_3inst = test.loc[test['label'].isin(['SD','HH','KD']),:]
train_3inst = train.loc[train['label'].isin(['SD','HH','KD']),:]
X_test_3inst=test_3inst.drop(['Unnamed: 0','label','filename','dirname_x','testOrTrain'],axis=1)
X_train_3inst=train_3inst.drop(['Unnamed: 0','label','filename','dirname_x','testOrTrain'],axis=1)
y_test_3inst= pd.DataFrame(test_3inst['label'])
y_train_3inst=pd.DataFrame(train_3inst['label'])

#ENST data only, split into testing (half of dry mix files) and training (all other files)
dryMix=dfENST_nomiss.loc[dfENST_nomiss['dirname_x']=='dry_mix',:]
fileListDry=dryMix.filename.unique()

#sort file list randomly, initialized with a seed to get the same result every time
random.seed(456)
random.shuffle(fileListDry)

#assign 50% of dry mix filenames to training, 50% to testing
isplit=round(len(fileListDry)*0.5)-1
filemapDry=[]

for i in range(0,len(fileListDry)):
    if i<=isplit:
        filemapDry.append([fileListDry[i],'training'])
    else:
        filemapDry.append([fileListDry[i],'testing'])

fileMapDryDF=pd.DataFrame(filemapDry,columns=['filename','testOrTrain'])

#merge in the training/testing column and split data
testTrainENST = dfENST_nomiss.merge(fileMapDryDF,how='inner',left_on=['filename'],right_on=['filename'])
dfENST_test=testTrainENST[(testTrainENST['testOrTrain'] == 'testing') & (testTrainENST['dirname_x']=='dry_mix')]
dfENST_train=testTrainENST[(testTrainENST['testOrTrain'] == 'training') | (testTrainENST['dirname_x']!='dry_mix')]
#n=len(dfENST_nomiss)
#dfENST_nomiss['rand']=np.random.randint(0,100,size=(n,1),dtype=int)
#dfENST_test=dfENST_nomiss.loc[(dfENST_nomiss['rand']<50) & (dfENST_nomiss['dirname_x']=='dry_mix'),:]
#dfENST_train=dfENST_nomiss.loc[(~dfENST_nomiss['rand']<50) | (dfENST_nomiss['dirname_x']=='dry_mix'),:]
X_test_ENST=dfENST_test.drop(['Unnamed: 0','label','filename','dirname_x'],axis=1)
X_train_ENST=dfENST_train.drop(['Unnamed: 0','label','filename','dirname_x'],axis=1)
y_test_ENST=pd.DataFrame(dfENST_test['label'])
y_train_ENST=pd.DataFrame(dfENST_train['label'])

#for combined MDB/ENST datasets, split into training (ESNT) and testing (MDB)
testAll=dfAll_nomiss[dfAll_nomiss.filename.str.contains('MusicDelta')]
trainAll=dfAll_nomiss[~dfAll_nomiss.filename.str.contains('MusicDelta')]

X_testAll= testAll.drop(['Unnamed: 0','label','filename','dirname_x'],axis=1)
X_trainAll=trainAll.drop(['Unnamed: 0','label','filename','dirname_x'],axis=1)
y_testAll= testAll['label']
y_trainAll=trainAll['label']


##########function to fit the SVM model for the given test and train data, feature list, and kernel type
def fitSVM(X_test,X_train,y_test,y_train,featureList,kern,deg):
    if featureList == 'All':
        X_train_sub=X_train
        X_test_sub=X_test
    else:
        X_train_sub=X_train[featureList]
        X_test_sub=X_test[featureList]
    print('predictors included=')
    for col in X_train_sub.columns:
        print (col)
    if deg!="":
        svclassifier = SVC(kernel=kern,degree=deg)
    else:
        svclassifier = SVC(kernel=kern)
    svclassifier.fit(X_train_sub, y_train)
    y_pred = svclassifier.predict(X_test_sub)
    print("features=",featureList)
    print("kernel=",kern)
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    rpt=classification_report(y_test,y_pred)
    print(rpt)
    F=float(str.split(rpt)[-2])
    return F,cm

##########list of baseline features
baseFeatures=['centroid','rms','zeroCrossings','crest','flux','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5']

print('Baseline: MDB Data only, 75/25 test/train split')
Flinear,cmLinear=fitSVM(X_test,X_train,y_test,y_train,featureList=baseFeatures,kern='linear',deg="")
Fsigmoid,cmSigmoid=fitSVM(X_test,X_train,y_test,y_train,featureList=baseFeatures, kern='sigmoid',deg="")
Fpoly2,cmPoly2=fitSVM(X_test,X_train,y_test,y_train,featureList=baseFeatures, kern='poly', deg=2)

#rerun SVM on MDB with 10 fold cross validation
#SVMcrossval returns average F score over the folds
#data=input containing all X's and all obs, featureList determines subset of X's, nFolds=number of folds
#kern=kernel shape, deg=degree (only applies if kern=poly)
def SVMcrossVal(data,featureList,kern,deg,nFolds):
    fileList=data.filename.unique()

    nInst=data.label.nunique()
    print('number of unique instruments=',nInst)

    #assign files to folds
    nFiles=len(fileList)
    baseSize=np.floor(nFiles/nFolds)
    remainder=nFiles-baseSize*nFolds
    print('nfiles=',nFiles,'baseSize=',baseSize,'remainder=',remainder)
    print('cutoff=',remainder*nFiles%(nFolds*baseSize))
    cutoff=remainder * nFiles % (nFolds * baseSize)
    foldList=np.zeros(nFiles)
    fold=0
    for i in range(0,nFiles):
        if i<cutoff:
            if i%remainder==0:
                fold+=1
        elif cutoff%2==0 and i%baseSize==0:
                fold+=1
        elif cutoff%2==1 and i%baseSize==1:
                fold+=1
        foldList[i]=fold

    foldmap=[(fileList[i],foldList[i]) for i,x in enumerate(fileList)]
    #print('foldmap=',foldmap)
    foldmapDF = pd.DataFrame(foldmap, columns=['filename', 'fold'])
    dataWfolds = data.merge(foldmapDF, how='inner', left_on=['filename'], right_on=['filename'])
    # for col in dataWfolds.columns:
    #     print (col)
    # print(dataWfolds)
    #for each fold, separate into testing and training, get the F score and confusion matrix from the SVM for that fold
    fScores=np.zeros(nFolds)
    #cms=np.zeros([nFolds,nInst,nInst])
    cms=[]
    print('initial cms',cms)
    for i in range (1,nFolds+1):
        X_train=dataWfolds[dataWfolds.fold!=i]
        X_test=dataWfolds[dataWfolds.fold==i]
        y_train=dataWfolds[dataWfolds.fold!=i]['label']
        y_test=dataWfolds[dataWfolds.fold==i]['label']
        print('processing fold ',i,'of ',nFolds)
        fScores[i-1],cm=fitSVM(X_test,X_train,y_test,y_train,featureList,kern,deg)
        cms.append(cm)
    print('fScores=',fScores)
    avg_f=np.average(fScores)
    #total_cm=cms.sum(axis=0)
    return avg_f,cms
avg_f,cms=SVMcrossVal(data=dfMDB_nomiss,featureList=baseFeatures,kern='sigmoid',deg="",nFolds=10)
###note: you can't sum up the confusion matrices over the folds because some folds do not contain all of the possible instrument classes, so the CM dimensions are different
print('average F score for MDB cross validation=',avg_f)
print('confusion matrices for MDB cross validation=',cms)

##MDB only, baseline feature list, 3 instrument classes
F,cm=fitSVM(X_test_3inst,X_train_3inst,y_test_3inst,y_train_3inst,featureList=baseFeatures, kern='sigmoid',deg="")
print('MDB data only, 3 instrument classes, baseline set of 10 features, sigmoid kernel')
print('F=',F)
print('confusion matrix=',cm)

##MDB+ENST data, baseline feature list, 3 instrument classes
F,cm=fitSVM(X_testAll,X_trainAll,y_testAll,y_trainAll,featureList=baseFeatures, kern='sigmoid',deg="")
print('All Data (ESNT+MDB), baseline set of 10 features, sigmoid kernel')
print('F=',F)
print('confusion matrix=',cm)

##ENST data only, baseline feature list
F,cm=fitSVM(X_test_ENST,X_train_ENST,y_test_ENST,y_train_ENST,featureList=baseFeatures, kern='sigmoid',deg="")
print('ENST data only, baseline set of 10 features, sigmoid kernel')
print('F=',F)
print('confusion matrix=',cm)
###################################################################################################
featureSetLogistic=['flux','mfcc1','mfcc2','mfcc3','mfcc4','flatness','rolloff','RMSF','lowRMS']

#MDB only. from the expanded pool of 29 features, use the 9 selected from logistic regression results
F,cm=fitSVM(X_test_3inst,X_train_3inst,y_test_3inst,y_train_3inst,featureList=featureSetLogistic, kern='sigmoid',deg="")
print('MDB data only, 3 instrument classes, set of 9 features from logistic reg, sigmoid kernel')
print('F=',F)
print('confusion matrix=',cm)

#ENST+MDB, 9 features selected from logistic regression results
F,cm=fitSVM(X_testAll,X_trainAll,y_testAll,y_trainAll,featureList=featureSetLogistic, kern='sigmoid',deg="")
print('All Data (ESNT+MDB), set of 9 features from logistic reg, sigmoid kernel')
print('F=',F)
print('confusion matrix=',cm)

#ENST data only, 9 features selected from logistic regression results
F,cm=fitSVM(X_test_ENST,X_train_ENST,y_test_ENST,y_train_ENST,featureList=featureSetLogistic, kern='sigmoid',deg="")
print('ENST data only, set of 9 features from logistic reg, sigmoid kernel')
print('F=',F)
print('confusion matrix=',cm)

##########################################################################################################
#from the expanded pool of 29 features, use the 9 selected from logistic regression results + 4 more identified as important from distribution plots
featureSetLogPlots=['flux','mfcc1','mfcc2','mfcc3','mfcc4','flatness','rolloff','RMSF','lowRMS','centroid','zeroCrossings','rms','crest']
F,cm=fitSVM(X_test_3inst,X_train_3inst,y_test_3inst,y_train_3inst,featureList=featureSetLogPlots, kern='sigmoid',deg="")
print('MDB data only, 3 instrument classes, set of 13 features from logistic reg and/or dist. plots, sigmoid kernel')
print('F=',F)
print('confusion matrix=',cm)

#ENST+MDB.
F,cm=fitSVM(X_testAll,X_trainAll,y_testAll,y_trainAll,featureList=featureSetLogPlots, kern='sigmoid',deg="")
print('All Data (ESNT+MDB), set of 13 features from logistic reg and/or dist. plots, sigmoid kernel')
print('F=',F)
print('confusion matrix=',cm)

#ENST data only, 13 features from logistic and/or distribution plots, sigmoid kernel
F,cm=fitSVM(X_test_ENST,X_train_ENST,y_test_ENST,y_train_ENST,featureList=featureSetLogPlots, kern='sigmoid',deg="")
print('ENST data only, set of 13 features from logistic reg and/or dist. plots, sigmoid kernel')
print('F=',F)
print('confusion matrix=',cm)

#MDB data only, 13 features from logistic and/or distribution plots, polynomial kernel
F,cm=fitSVM(X_test_3inst,X_train_3inst,y_test_3inst,y_train_3inst,featureList=featureSetLogPlots, kern='poly',deg=4)
print('ENST data only, set of 13 features from logistic reg and/or dist. plots, poly kernel deg 4')
print('F=',F)
print('confusion matrix=',cm)

#ENST+MDB, 13 features, polynomial kernel
F,cm=fitSVM(X_testAll,X_trainAll,y_testAll,y_trainAll,featureList=featureSetLogPlots, kern='poly',deg=4)
print('All Data (ESNT+MDB), set of 13 features from logistic reg and/or dist. plots, poly kernel deg 4')
print('F=',F)
print('confusion matrix=',cm)

#ENST data only, 13 features, polynomial kernel
F,cm=fitSVM(X_test_ENST,X_train_ENST,y_test_ENST,y_train_ENST,featureList=featureSetLogPlots, kern='poly',deg=4)
print('ENST data only, set of 13 features, polynomial degree 4 kernel')
print('F=',F)
print('confusion matrix=',cm)