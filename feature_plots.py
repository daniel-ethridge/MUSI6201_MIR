#program: feature_plots.py
#purpose: plot the pdf of the features by instrument for all data combined

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dirname='C:/Users/Laney/Documents/6201_comp_mus_analysis/Final project/data/'
filename='allDataNorm.csv'
datadf = pd.read_csv(dirname+filename)

instruments=['SD','HH','KD','CY']
for col in datadf.columns:
    print (col)
featureList=['centroid','rms','zeroCrossings','crest','flux','mfcc1','mfcc2','mfcc3','mfcc4',
'mfcc5','flatness','rolloff','c0','c100','c200','c400','c800','c1600','c3200','RMSF','lowRMS','bandRMS',
'highRMS','lowRMSrel','bandRMSrel','highRMSrel','lowRMSrelband','lowRMSrelhigh','bandRMSrelhigh']

nFeatures=len(featureList)
print('nFeatures=',nFeatures)

i=1

plt.rcParams.update({'font.size': 22})
for feature in featureList:
    for inst in instruments:
        data=datadf[datadf['label']==inst][feature].dropna()
        sns.distplot([data],hist=False,label=inst)
    if i==1:
        plt.legend(title='Instrument')
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.show()
    i+=1

