from scipy.signal import butter, sosfilt
from scipy.io.wavfile import read as wavread
import numpy as np

cAudioFilePath = ""

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

def block_audio(x, blockSize, hopSize, fs):
    # allocate memory
    numBlocks = np.ceil(x.size / hopSize).astype('int')
    xb = np.zeros([numBlocks, blockSize])

    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)

    for n in range(numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]

    return xb, t

def FeatureTimeRms(xb):    
    """From Alexander's assignment 2 reference code"""
    # number of results    
    numBlocks = xb.shape[0] 
    
    # allocate memory    
    vrms = np.zeros(numBlocks)    
    
    for n in range(0, numBlocks):        
        # calculate the rms        
        vrms[n] = np.sqrt(np.dot(xb[n,:], xb[n,:]) / xb.shape[1])    
        
    # convert to dB    
    epsilon = 1e-5  # -100dB    
    vrms[vrms < epsilon] = epsilon    
    vrms = 20 * np.log10(vrms)    
        
    return (vrms)

def butter_bandpass(lowpass, highpass, fs, order=5):
        nyq = 0.5 * fs
        low = lowpass / nyq
        high = highpass / nyq
        lowsos = butter(order, low, analog=False, btype='low', output='sos')
        bandsos = butter(order, [low, high], analog=False, btype='band', output='sos')
        highsos = butter(order, high, analog=False, btype='high', output='sos')
        
        return lowsos, bandsos, highsos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        [lowsos, bandsos, highsos] = butter_bandpass(lowcut, highcut, fs, order=order)
        
        lowy = sosfilt(lowsos, data)
        bandy = sosfilt(bandsos, data)
        highy = sosfilt(highsos, data)
        return lowy, bandy, highy
    
def filtered_RMS(cAudioFilePath, lowcut=200, highcut=1000, blockSize=1024, hopSize=512):
    
    fs, audio = ToolReadAudio(cAudioFilePath)
    [lowy, bandy, highy] = butter_bandpass_filter(audio, lowcut, highcut, fs)
    
    xb, _ = block_audio(audio, blockSize, hopSize, fs)
    lowxb, _ = block_audio(lowy, blockSize, hopSize, fs)
    bandxb, _ = block_audio(bandy, blockSize, hopSize, fs)
    highxb, _ = block_audio(highy, blockSize, hopSize, fs)
    
    RMS = FeatureTimeRms(xb)
    lowRMS = FeatureTimeRms(lowxb)
    bandRMS = FeatureTimeRms(bandxb)
    highRMS = FeatureTimeRms(highxb)
    lowRMSrel = lowRMS - RMS
    bandRMSrel = bandRMS - RMS
    highRMSrel = highRMS - RMS
    lowRMSrelband = lowRMS - bandRMS
    lowRMSrelhigh = lowRMS - highRMS
    bandRMSrelhigh = bandRMS - highRMS
    
    RMSmatrix = np.stack((RMS, lowRMS, bandRMS, highRMS, lowRMSrel, bandRMSrel, highRMSrel, 
            lowRMSrelband, lowRMSrelhigh, bandRMSrelhigh))
    
    return RMSmatrix


RMSmatrix = filtered_RMS(cAudioFilePath, blockSize=1024, hopSize=512)
