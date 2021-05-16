from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)
import numpy as np
import MFCC
from sklearn.model_selection import train_test_split

rootdir='C:/Users/Laney/Documents/6201_comp_mus_analysis/Final project/data/MDB Drums'

#calculate MFCC for all audio files
MFCC_dict = calcMFCC(dir=rootdir+'/audio/drum_only')  ##should we be using the full audio files instead?

#get drum class from annotated data
dirname = rootdir+'/annotations/class'
classdict = {}
for fn in os.listdir(dirname):
    basename = fn.replace('_class.txt',"")
    anno = open(dirname + '/' + fn ,'r')
    classtimes = []
    for line in anno.readlines():
        classtime = float(line.split()[0])
        classtimes.append(classtime)
    print('bt ',fn,len(classtimes))
    classdict[basename] = classtimes

#split into training (75%) and testing (25%) datasets
#how do we make sure a wav file doesn't get split between the training and testing?
X_train, X_test, y_train, y_test = train_test_split(MFCC_dict, classdict, stratify=classdict, test_size=0.25, random_state=123)

nca = NeighborhoodComponentsAnalysis(random_state=123)
knn = KNeighborsClassifier(n_neighbors=3)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
nca_pipe.fit(X_train, y_train)

print(nca_pipe.score(X_test, y_test))