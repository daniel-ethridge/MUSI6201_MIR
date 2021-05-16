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
    if len(onsets) == 0:
        onsets = np.zeros(1)
    return onsets

# def create_compound_labels(df):
#     print("ccl ",df)
#     return df
# labs = ['CY','HH','KD','SD']
# all_labs = ['x'] * 16
# lab_dict = {}
# print("befor ",all_labs)
# curr_index = 0
# for x in range(1,5):
#     compound_labs = [",".join(map(str,comb)) for comb in combinations(labs,x)]
#     for y in range(len(compound_labs)):
#         print("y ",y,curr_index)
#         lab_dict[compound_labs[y]] = curr_index
#         curr_index += 1
#print("all labs ",lab_dict)
def get_labels(basename,anno,beatdict,labelmatrix):
    beattimes = []
    for line in anno.readlines():
        beattime = float(line.split()[0])
        
        beatlabel = line.split()[1]
        beattimes.append([beattime,beatlabel])
        labelmatrix.append([basename,beattime,'Label',beatlabel])
    print('bt ',anno,len(beattimes),beattimes[-1][0]/len(beattimes),beattimes[-1][0])
    beatdict[basename] = np.asarray(beattimes)
    return beatdict,labelmatrix

def get_beat_offsets(basename,dirname,full_fn,offdf,offdict):    
    try:
        x_raw,fs = madmom.io.audio.load_wave_file(full_fn)
    except Exception as badfile:
        print("bad file ", badfile, full_fn)
        return #(pd.DataFrame([0]),pd.DataFrame([0],dict('y'=0)
    # print("full_fn",full_fn)
    # last_slash = full_fn.rfind('/')
    # basename = full_fn[last_slash + 1:].replace('.wav',"")
    # print("bn",basename)
    #x = 1 / 0
    if x_raw.ndim > 1:
        x = madmom.audio.signal.remix(x_raw,1)
    else:
        x = x_raw
    blockSize = 4096
    hopSize = 2048
    user_time_window = float(hopSize) / float(fs)
    beats,winvector = get_window_beat_vector(full_fn,user_time_window,beatoronset='onset')
    offdict[basename] = beats
    offsetdf = pd.DataFrame(beats.reshape(-1),columns=['time'])
    offsetdf['filename'] = basename
    offsetdf['dirname'] = dirname
    offdf = pd.concat([offdf,offsetdf],sort=False)
    offdf['type'] = 'Onset'
    return offdf,offdict

def get_signal_features(basename,dirname,full_fn,feat_df,offdf,offdict,featdict,mfccdict):
    blockSize = 4096
    hopSize = 2048
    user_time_window = float(hopSize) / float(fs)
    feat_mat = ass2.extract_features(x,blockSize,hopSize,fs)
    sig_df = pd.DataFrame(feat_mat.reshape((feat_mat.shape[1],feat_mat.shape[0])),columns=['centroid', 'rms', 'zeroCrossings', 'crest', 'flux'])
    #print('fdf ',feat_df.head())
    y, sr = librosa.load(full_fn)
    mfcc_mat = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=5,hop_length=int(hopSize/2))
    mfcc_df = pd.DataFrame(mfcc_mat.reshape((mfcc_mat.shape[1],mfcc_mat.shape[0])),columns=['mfcc1','mfcc2','mfcc3','mfcc4','mfcc5'])
    flatness_mat = librosa.feature.spectral_flatness(y=y,hop_length=int(hopSize/2))
    #print(flatness_mat.shape)
    mfcc_df['flatness'] = flatness_mat.reshape(-1)
    rolloff_mat=librosa.feature.spectral_rolloff(y=y,sr=sr,hop_length=int(hopSize/2))
    mfcc_df['rolloff'] = rolloff_mat.reshape(-1)
    contrast_mat=librosa.feature.spectral_contrast(y=y,sr=sr,fmin=50,hop_length=int(hopSize/2))
    contrast_df = pd.DataFrame(contrast_mat.reshape((contrast_mat.shape[1],contrast_mat.shape[0])),columns=['c0','c100','c200','c400','c800','c1600','c3200'])
    mfcc_df = pd.concat([mfcc_df,contrast_df],axis=1).reset_index()
    sig_df = pd.concat([sig_df,mfcc_df],axis=1).reset_index()
    sig_df['filename'] = basename
    sig_df['dirname'] = dirname
    sig_df['type'] = 'Sample'
    #print("user_time_window ",sig_df.shape[0]*user_time_window)
    feat_times = pd.Series(np.linspace(0,user_time_window*sig_df.shape[0],num=sig_df.shape[0]))
    #print("feat_times ",feat_times.head())
    sig_df['time'] = feat_times
    feat_df = pd.concat([feat_df,sig_df],sort=False)
    #print("final size ",feat_df.head(),feat_df.shape) 
    
    #print("offdf ",offdf.head(),offdf.shape)
    featdict[basename] = feat_mat
    mfccdict[basename] = mfcc_mat
    #print('ot ', fn, len(beats))
    #otlen += len(beats)
    return feat_df,featdict,mfccdict


def get_label_data():
    dirname = '/Users/bclark66/Downloads/MDBDrums-master/MDB Drums/annotations/class'
    beatdict = {}
    
    btlen = 0
    labelmatrix = []
    for fn in os.listdir(dirname):
        basename = fn.replace('_class.txt',"")
        print("beatdict ",basename)
        anno = open(dirname + '/' + fn ,'r')
        beatdict,labelmatrix = get_labels(basename,anno,beatdict,labelmatrix)

    dirnames = ['/Users/bclark66/Downloads/ENST-drums-public/drummer_1/annotation',
            '/Users/bclark66/Downloads/ENST-drums-public/drummer_2/annotation',
            '/Users/bclark66/Downloads/ENST-drums-public/drummer_3/annotation']
    for dirname in dirnames:
        for fn in os.listdir(dirname):
            if fn[0] == '.':
                continue
            basename = fn.replace('.txt',"")
            anno = open(dirname + '/' + fn ,'r')
            beatdict,labelmatrix = get_labels(basename,anno,beatdict,labelmatrix)
    beatdf = pd.DataFrame(labelmatrix,columns=['filename','time','type','label']).sort_values(['filename','time','type','label']).reset_index(drop=True)
    beatdf['dirname'] = 'annotation'
    return beatdict,labelmatrix,beatdf
    


    #btlen += len(beattimes)
def get_signal_data(offsets_only=False):
    #print("btlen ",btlen,labelmatrix)
    
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
        offdf,offdict = get_beat_offsets(basename,dirname,full_fn,offdf,offdict)
        if offsets_only:
            pass
        else:
            feat_df,featdict,mfccdict = get_signal_features(basename,dirname,full_fn,feat_df,featdict,mfccdict)

    dirnames = ['/Users/bclark66/Downloads/ENST-drums-public/drummer_1/audio',
            '/Users/bclark66/Downloads/ENST-drums-public/drummer_2/audio',
            '/Users/bclark66/Downloads/ENST-drums-public/drummer_3/audio']
    sub_dirnames = ['accompaniment','dry_mix','hi-hat','kick','overhead_L','overhead_R',
                    'snare','tom_1','tom_2','wet_mix']
    for dirname in dirnames:
        for sub_dir in sub_dirnames:
            full_dirname = dirname + '/' + sub_dir
            for fn in os.listdir(full_dirname):
                if fn[0] == '.':
                    continue
                basename = fn.replace('.wav','')
                full_fn = full_dirname + '/' + fn
                offdf,offdict = get_beat_offsets(basename,sub_dir,full_fn,offdf,offdict)
                if offsets_only:
                    pass
                else:
                    feat_df,featdict,mfccdict = get_signal_features(basename,sub_dir,full_fn,feat_df,featdict,mfccdict)
    offdf = offdf.sort_values(['filename','dirname','time']).reset_index(drop=True)
    offdf.to_csv("offsets.csv")
    if offsets_only:
        return
    feat_df = feat_df.sort_values(['filename','dirname','time']).reset_index(drop=True)
    
    feat_df = feat_df.merge(offdf,how='outer',right_on=['filename','dirname','time','type'],left_on=['filename','dirname','time','type']).reset_index(drop=True)
    #feat_df = feat_df.fillna(method='bfill')
    feat_df = feat_df.sort_values(['filename','time']).reset_index(drop=True)
    feat_df = feat_df.merge(beatdf,how='outer',right_on=['filename','time','type'],left_on=['filename','time','type']).reset_index(drop=True)
    #feat_df = feat_df.fillna(method='bfill')
    feat_df = feat_df.sort_values(['filename','time']).reset_index(drop=True)
    #feat_df = feat_df.fillna(method='bfill')
    # print("fdf 2 ",feat_df.head)
    feat_df.to_csv("test output.csv")
    return featdf,offdf,offdict,featdict,mfccdict
    #print("Total onsets ",otlen)
def label_data(beatdict,offdict,featdict,mfccdict):   
    labeled_data = []  
    grand_total = 0
    user_time_window = .40
    for bn in beatdict.keys():
        print("labeling ",bn)
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
                    labeled_data_row.append('annotation')
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
    
    datadf = pd.DataFrame(labeled_data,columns=['filename','time','window','centroid', 'rms', 'zeroCrossings', 'crest', 'flux','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','dirname','label'])
    datadf = datadf.fillna(0)
    datadf.sort_values(['filename','window','label'],inplace=True)
    datadf2 = datadf.groupby(['filename','window'])['label'].apply(','.join).reset_index()
    datadf = datadf.merge(datadf2,how='inner',left_on=['filename','window'],right_on=['filename','window'])
    # print('bdf ',beatdf)
    # print('off ',offdf)
    # print("data ",datadf.head())
    datadf.to_csv('ACA Project Data.csv')
    return datadf

get_signal_data(offsets_only=True)

datadf = pd.read_csv('ACA Project Data.csv')
#feat_df,datadf = get_raw_data()
#combine sd and SD
datadf.loc[datadf['label_x'].isin(['sd','SD','-sd']),['label_x']] = 'SD'
datadf.loc[datadf['label_x'].isin(['chh','HH']),['label_x']] = 'HH'
datadf.loc[datadf['label_x'].isin(['KD','bd']),['label_x']]  = 'KD'
datadf.loc[datadf['label_x'].isin(['CY','rc3','rc4','ohh',]),['label_x']]  = 'CY'
datadf = datadf.loc[datadf['label_x'].isin(['SD','HH','KD']),:]

X = datadf.drop(['filename','time','window','label_x','label_y','dirname','zeroCrossings','crest','mfcc2','mfcc3','mfcc4','mfcc5'],axis=1)
y = datadf['label_x']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#svclassifier = SVC(kernel='linear')
svclassifier = SVC(kernel='poly',degree=4)
#svclassifier = SVC(kernel='rbf')
#svclassifier = SVC(kernel='sigmoid')

svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("y_pred",y_pred)
cm = confusion_matrix(y_test,y_pred)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)

cax = ax.matshow(cm) #,cmap='summer')
fig.colorbar(cax)
ax.set_xticklabels(sorted(list(set(y_pred)),reverse=True))
ax.set_yticklabels(sorted(list(set(y_pred)),reverse=True))
plt.show()
print(classification_report(y_test,y_pred))





