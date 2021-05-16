# Program name: norm_feature_data.py
# Purpose: normalize the feature data and group the instruments from ENST Drums into the same categories used in MDB
# Output datasets to CSV for 1) MDB data only, 2) both datasets combined, 3) ENST data only

import pandas as pd
from sklearn.preprocessing import StandardScaler
dirname='C:/Users/Laney/Documents/6201_comp_mus_analysis/Final project/data/'
filename='feat_df4.csv'

datadf = pd.read_csv(dirname+filename)
datadf.describe

MDBdf=datadf[datadf['filename'].str.contains('MusicDelta')]

datadf.loc[datadf['label'].isin(['sd','SD','-sd']),['label']] = 'SD'
datadf.loc[datadf['label'].isin(['chh','HH']),['label']] = 'HH'
datadf.loc[datadf['label'].isin(['KD','bd']),['label']]  = 'KD'
datadf.loc[datadf['label'].isin(['CY','rc3','rc4','ohh',]),['label']]  = 'CY'
datadf = datadf.loc[datadf['label'].isin(['SD','HH','KD']),:]
datadf=datadf.loc[datadf['type']=='Sample',:]

ENSTdf=datadf[~datadf['filename'].str.contains('MusicDelta')]

XAll = datadf.drop(['Unnamed: 0','Unnamed: 0.1','index','filename','time','label','dirname_x','type','time','label','dirname_y','anno_time_window_low','anno_time_window_hi'],axis=1)
yAll = datadf[['label','filename','dirname_x']]

XMDB = MDBdf.drop(['Unnamed: 0','Unnamed: 0.1','index','filename','time','label','dirname_x','type','time','label','dirname_y','anno_time_window_low','anno_time_window_hi'],axis=1)
yMDB = MDBdf[['label','filename','dirname_x']]

XENST = ENSTdf.drop(['Unnamed: 0','Unnamed: 0.1','index','filename','time','label','dirname_x','type','time','label','dirname_y','anno_time_window_low','anno_time_window_hi'],axis=1)
yENST = ENSTdf[['label','filename','dirname_x']]

#normalize features across both datasets combined
scaler = StandardScaler()
scaler.fit(XAll)
XnormAll=scaler.transform(XAll)
XnormAll=pd.DataFrame(XnormAll, columns=XAll.columns)

#normalize features across MDB only
scaler.fit(XMDB)
XnormMDB=scaler.transform(XMDB)
XnormMDB=pd.DataFrame(XnormMDB, columns=XMDB.columns)

#normalize features across ENST only
scaler.fit(XENST)
XnormENST=scaler.transform(XENST)
XnormENST=pd.DataFrame(XnormENST, columns=XENST.columns)

#add filename and labels column back into the normalized X data
XnormAll.reset_index(drop=True,inplace=True)
yAll.reset_index(drop=True,inplace=True)
allDataNorm=pd.concat([XnormAll,yAll], axis=1)

XnormMDB.reset_index(drop=True,inplace=True)
yMDB.reset_index(drop=True,inplace=True)
MDBdataNorm=pd.concat([XnormMDB,yMDB], axis=1)

XnormENST.reset_index(drop=True,inplace=True)
yENST.reset_index(drop=True,inplace=True)
ENSTdataNorm=pd.concat([XnormENST,yENST], axis=1)


if __name__ == "__main__":
    dt='12_4_19'
    allDataNorm.to_csv(dirname+'allDataNorm'+dt+'.csv')
    MDBdataNorm.to_csv(dirname+'MDBdataNorm'+dt+'.csv')
    ENSTdataNorm.to_csv(dirname + 'ENSTdataNorm'+dt+'.csv')

