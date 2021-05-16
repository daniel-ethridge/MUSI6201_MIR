from scipy.io.wavfile import read as wavread
import librosa.feature as lib
import os

dirname = '/Users/ethri/Desktop/MIR/MIR_Project/MDBDrums/MDB Drums/audio/drum_only'

def ToolReadAudio(cAudioFilePath):
    [samplerate, x] = wavread(cAudioFilePath)
    if x.dtype == 'float32':
        audio = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32
        audio = x / float(2**(nbits - 1))
    # special case of unsigned format
    if x.dtype == 'uint8':
        audio = audio - 1.
    return (samplerate, audio)


def MFCC(dirname):
    MFCC_dict = {}
    for fn in os.listdir(dirname):
        samplerate, audio = ToolReadAudio(str(dirname + '/' + fn))
        MFCC = lib.mfcc(audio, samplerate)
        MFCC_dict['{0}'.format(fn)] = MFCC

    return MFCC_dict

MFCC_dict = MFCC(dirname)
print(MFCC_dict)
