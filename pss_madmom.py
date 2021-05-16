import madmom
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import math
import pandas as pd
import ass2solution_2 as ass2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from itertools import combinations


def get_window_beat_vector(fn,user_time_window,beatoronset='beat'):
    
    if beatoronset == 'beat':
        beats = get_beats(fn)
    else:
        beats = get_onsets(fn)
    wvlen = int(beats[-1]/user_time_window) + 1
    winvector = [0  * i for i in range(wvlen)]
    for beatnbr in range(len(beats)):
        nextwindow = int(beats[beatnbr] // user_time_window) 
        winvector[nextwindow] = 1
    return beats,winvector

def get_beats(fn):
    act = madmom.features.beats.RNNBeatProcessor()(fn)
        
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    beats = proc(act)
    return beats

def get_onsets(fn):
    proc = madmom.features.onsets.OnsetPeakPickingProcessor(fps=100)
    act = madmom.features.onsets.RNNOnsetProcessor()(fn)
    onsets = proc(act)
    return onsets

def create_compound_labels(df):
    print("ccl ",df)
    return df
labs = ['CY','HH','KD','SD']
all_labs = ['x'] * 16
lab_dict = {}
print("befor ",all_labs)
curr_index = 0
for x in range(1,5):
    compound_labs = [",".join(map(str,comb)) for comb in combinations(labs,x)]
    for y in range(len(compound_labs)):
        print("y ",y,curr_index)
        lab_dict[compound_labs[y]] = curr_index
        curr_index += 1
print("all labs ",lab_dict)
dirname = '/Users/bclark66/Downloads/MDBDrums-master/MDB Drums/annotations/class'
beatdict = {}
user_time_window = .10
btlen = 0
labelmatrix = []
for fn in os.listdir(dirname):
    basename = fn.replace('_class.txt',"")
    anno = open(dirname + '/' + fn ,'r')
    beattimes = []
    for line in anno.readlines():
        beattime = float(line.split()[0])
        
        beatlabel = line.split()[1]
        beattimes.append([beattime,beatlabel])
        labelmatrix.append([basename,beattime,'Label',beatlabel])
    print('bt ',fn,len(beattimes),beattimes[-1][0]/len(beattimes),beattimes[-1][0])
    beatdict[basename] = np.asarray(beattimes)
    btlen += len(beattimes)

#print("btlen ",btlen,labelmatrix)
beatdf = pd.DataFrame(labelmatrix,columns=['filename','time','type','label']).sort_values(['filename','time','type','label']).reset_index(drop=True)
#print('bdf ',beatdf.head())
#print("beatdict",beatdict)
offdict = {}
featdict = {}
mfccdict = {}
#dirname = '/Users/bclark66/Downloads/MDBDrums-master/MDB Drums/audio/drum_only'
dirname = '/Users/bclark66/Downloads/MDBDrums-master/MDB Drums/audio/full_mix'
labeldict = {}
otlen = 0
feat_df = pd.DataFrame()
offdf = pd.DataFrame()
for fn in os.listdir(dirname):
    #basename = fn.replace('_Drum.wav','')
    basename = fn.replace('_MIX.wav','')
    full_fn = dirname + '/' + fn
    
    x_raw,fs = madmom.io.audio.load_wave_file(full_fn)
    #print("fs ",fs,x_raw.shape)
    if x_raw.shape[1] > 1:
        x = madmom.audio.signal.remix(x_raw,1)
    else:
        x = x_raw
    blockSize = 1024
    hopSize = 512
    user_time_window = float(hopSize) / float(fs)
    feat_mat = ass2.extract_features(x,blockSize,hopSize,fs)
    sig_df = pd.DataFrame(feat_mat.reshape((feat_mat.shape[1],feat_mat.shape[0])),columns=['centroid', 'rms', 'zeroCrossings', 'crest', 'flux'])
    #print('fdf ',feat_df.head())
    y, sr = librosa.load(full_fn)
    mfcc_mat = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=5,hop_length=int(hopSize/2))
    mfcc_df = pd.DataFrame(mfcc_mat.reshape((mfcc_mat.shape[1],mfcc_mat.shape[0])),columns=['mfcc1','mfcc2','mfcc3','mfcc4','mfcc5'])
    sig_df = pd.concat([sig_df,mfcc_df],axis=1).reset_index()
    sig_df['filename'] = basename
    sig_df['type'] = 'Sample'
    #print("user_time_window ",sig_df.shape[0]*user_time_window)
    feat_times = pd.Series(np.linspace(0,user_time_window*sig_df.shape[0],num=sig_df.shape[0]))
    #print("feat_times ",feat_times.head())
    sig_df['time'] = feat_times
    feat_df = pd.concat([feat_df,sig_df],sort=False)
    #print("final size ",feat_df.head(),feat_df.shape)
    
    
    beats,winvector = get_window_beat_vector(full_fn,user_time_window,beatoronset='onset')
    offdict[basename] = beats
    offsetdf = pd.DataFrame(beats.reshape(-1),columns=['time'])
    offsetdf['filename'] = basename
    offdf = pd.concat([offdf,offsetdf],sort=False)
    offdf['type'] = 'Onset'
    #print("offdf ",offdf.head(),offdf.shape)
    featdict[basename] = feat_mat
    mfccdict[basename] = mfcc_mat
    print('ot ', fn, len(beats))
    otlen += len(beats)

print("Total onsets ",otlen)
   
labeled_data = []  
grand_total = 0
for bn in beatdict.keys():
    correct = 0
    total = 0
    #print("slsls",bn,beatdict[bn],offdict[bn])
    for beat in range(len(beatdict[bn])):
        #print("gt ",beatdict[bn][beat])
        firstbeat = float(beatdict[bn][beat][0])
        total +=1
        grand_total +=1
        for onset in range(len(offdict[bn])):
            labeled_data_row = []
            thisonset = float(offdict[bn][onset])
            if  thisonset >= firstbeat - user_time_window and thisonset <= firstbeat + user_time_window:
                correct += 1
                label = beatdict[bn][beat][1]
                labeled_data_row.append(bn)
                labeled_data_row.append(firstbeat)
                labeled_data_row.append(thisonset)
                for feat in featdict[bn][:,onset]:
                    labeled_data_row.append(feat)
                for mfcc in mfccdict[bn][:,onset]:
                    labeled_data_row.append(mfcc)
                labeled_data_row.append(label)
                labeled_data.append(labeled_data_row)
            
    #print("data ",labeled_data)         
    correct_positives = correct
    false_negatives = total - correct
    correct = 0
    total = 0
    for onset in range(len(offdict[bn])):
        thisonset = float(offdict[bn][onset])
        total +=1
        for beat in range(len(beatdict[bn][0])):
            firstbeat = float(beatdict[bn][beat][0])
            if  thisonset >= firstbeat - user_time_window and thisonset <= firstbeat + user_time_window:
                correct += 1
    false_positives = total - correct            
    print("bn ",bn,'total ',total,"fn ",false_negatives,"fp ",false_positives,"cp ",correct_positives," precision ",
        correct_positives/(correct_positives + false_negatives),"recall ",
        correct_positives/(correct_positives + false_positives))
print("grand_total ",grand_total,len(labeled_data))


#beatdf = pd.DataFrame([beatdict]) #,columns=['filename','time','label'])
# label_window = beatdf.groupby(['filename','time'])['time'].diff().mean()
# print("label_window" ,label_window)


#offdf = pd.DataFrame([offdict])
# print("offdf ",offdf.head())
# print("feat_df ",feat_df.head())
feat_df = feat_df.sort_values(['filename','time']).reset_index(drop=True)
offdf = offdf.sort_values(['filename','time']).reset_index(drop=True)
feat_df = feat_df.merge(offdf,how='outer',right_on=['filename','time','type'],left_on=['filename','time','type']).reset_index(drop=True)
#feat_df = feat_df.fillna(method='bfill')
feat_df = feat_df.sort_values(['filename','time']).reset_index(drop=True)
feat_df = feat_df.merge(beatdf,how='outer',right_on=['filename','time','type'],left_on=['filename','time','type']).reset_index(drop=True)
#feat_df = feat_df.fillna(method='bfill')
feat_df = feat_df.sort_values(['filename','time']).reset_index(drop=True)
#feat_df = feat_df.fillna(method='bfill')
# print("fdf 2 ",feat_df.head)
feat_df.to_csv("test output.csv")
datadf = pd.DataFrame(labeled_data,columns=['filename','time','window','centroid', 'rms', 'zeroCrossings', 'crest', 'flux','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','label'])
datadf = datadf.fillna(0)
datadf.sort_values(['filename','window','label'],inplace=True)
datadf2 = datadf.groupby(['filename','window'])['label'].apply(','.join).reset_index()
datadf = datadf.merge(datadf2,how='inner',left_on=['filename','window'],right_on=['filename','window'])
# print('bdf ',beatdf)
# print('off ',offdf)
# print("data ",datadf.head())
datadf.to_csv('ACA Project Data.csv')
X = datadf.drop(['filename','time','window','label_x','label_y'],axis=1)
y = datadf['label_y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#svclassifier = SVC(kernel='linear')
#svclassifier = SVC(kernel='poly',degree=8)
#svclassifier = SVC(kernel='rbf')
svclassifier = SVC(kernel='sigmoid')

svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)

cax = ax.matshow(cm) #,cmap='summer')
fig.colorbar(cax)
ax.set_xticklabels([' ','CY','HH','KD','OT','SD','TT'])
ax.set_yticklabels([' ','CY','HH','KD','OT','SD','TT'])
plt.show()
print(classification_report(y_test,y_pred))





