#############################
#prelim_analysis.py
#Purpose: preliminary analysis of annotation data in MDB drums
#############################
import os
import pandas as pd
import numpy as np

rootdir='C:/Users/Laney/Documents/6201_comp_mus_analysis/Final project/data/MDB Drums'

fileNum=1
data=[]
dirname=rootdir + '/annotations/class/'
for fn in os.listdir(dirname):
    onsetNum=1
    anno = open(dirname + fn, 'r')
    basename = fn.replace('_class.txt', "")
    for line in anno.readlines():
        onsetTime = line.split()[0]
        instrument = line.split()[1]
        data.append([fileNum, onsetNum, basename, onsetTime, instrument])
        onsetNum=onsetNum+1
    fileNum = fileNum + 1

#convert to pandas dataframe
df=pd.DataFrame(data, columns=['fileNum','onsetNum','fileName','onsetTime','instrument'])
df['onsetTime']=df['onsetTime'].astype(float)
df[100:120]

#add lag of onsetTime
df['lagTime']=df.groupby('fileName').onsetTime.shift(1).fillna(0)
df['lagFileNum']=df.groupby('fileName').fileNum.shift(1).fillna(0)

#calculate time between current and last annotated onset
df['timeDiff']=df['onsetTime'] - df['lagTime']

#define whether the row represents a new onset, or if it can be grouped together with the previous onset because they are less than the window length apart
window=512/44100 # approx. 11 ms
df.loc[(df.timeDiff<window) & (df.lagFileNum!=0),'newOnset']=0
df.loc[(df.timeDiff>=window) | (df.lagFileNum ==0),'newOnset']=1

df['groupedOnsetNum']=df.groupby('fileName').newOnset.cumsum()

#print sample of data to check logic and format
df[100:120]

print ('N drum onsets in entire dataset=',len(data))

print ('frequency table of all instruments in annotation file')
df.instrument.value_counts()

# print ('count all occurrences of each instrument in the annotation files, grouped by exact onset time')
pivotTable=df.pivot_table(index=['fileNum','fileName','onsetTime'],columns='instrument',aggfunc=lambda x: 1,fill_value=0)['onsetNum']
pivotTable[0:10]
print ('pattern matrix of instruments occurring at the exact same onset time')
pivotTable.groupby(["CY", "HH", "KD", "OT", "SD", "TT"]).size().reset_index()


pivotTable2=df.pivot_table(index=['fileNum','fileName','groupedOnsetNum'],columns='instrument',aggfunc=lambda x: 1,fill_value=0)['onsetNum']
pivotTable2[0:10]
print ('pattern matrix of instruments occurring within same window')
pivotTable2.groupby(["CY", "HH", "KD", "OT", "SD", "TT"]).size().reset_index()

#####################
# #version 2: only look at 3 most common instruments
# fileNum=1
# dataV2=[]
# for fn in os.listdir(dirname):
#     onsetNum=1
#     anno = open(rootdir + '/annotations/class/' + fn, 'r')
#     basename = fn.replace('_class.txt', "")
#     for line in anno.readlines():
#         onsetTime = line.split()[0]
#         instrument = line.split()[1]
#         if (instrument=='SD' | instrument=='HH' | instrument=='KD'):
#             data.append([fileNum, onsetNum, basename, onsetTime, instrument])
#         onsetNum=onsetNum+1
#     fileNum = fileNum + 1