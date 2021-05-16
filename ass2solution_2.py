import numpy as np
import scipy.signal as sig
from scipy.io.wavfile import write, read
import glob
import os
from matplotlib import pyplot as plt


def wavread(filename):
    INT16_FAC = (2**15)-1
    INT32_FAC = (2**31)-1
    INT64_FAC = (2**63)-1
    norm_fact = {'int16': INT16_FAC, 'int32': INT32_FAC, 'int64': INT64_FAC, 'float32': 1.0, 'float64': 1.0}
    
    fs, x = read(filename)
    x = np.float32(x)/norm_fact[x.dtype.name]

    return fs, x



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



def extract_spectral_centroid(xb, fs):
    blockSize = xb.shape[1]
    hannWindow = sig.hann(blockSize, True)

    spectralCentroids = []
    for block in xb:
        block *= hannWindow
        blockFFT = np.abs(np.fft.fft(block)[:blockSize//2])        
        spectralCentroids.append(np.sum(np.arange(blockFFT.size) * blockFFT) / np.sum(blockFFT))

    return np.array(spectralCentroids) * fs / (blockSize)


        
def extract_rms(xb):
    RMS = []    
    for block in xb:
        blockRMS = np.sqrt(np.mean(block**2))
        RMSdB = 20 * np.log10(blockRMS)
        RMSdB = np.clip(RMSdB, a_min=-100, a_max=None)
        RMS.append(RMSdB)
    return np.array(RMS)


    
def extract_zerocrossingrate(xb):
    blockSize = xb.shape[1]
    zerocrossings = []
    
    for block in xb:    
        zerocrossings.append(np.sum(np.abs(np.diff(np.sign(block))) / (2*blockSize)))

    return np.array(zerocrossings)


    
def extract_spectral_crest(xb):
    blockSize = xb.shape[1]
    hannWindow = sig.hann(blockSize, True)
    spectralCrests = []
    for block in xb:    
        block *= hannWindow
        blockFFT = np.abs(np.fft.fft(block)[:blockSize//2])        
        spectralCrests.append(np.max(blockFFT) / np.sum(blockFFT))
    return np.array(spectralCrests)
    

    
def extract_spectral_flux(xb):
    blockSize = xb.shape[1]
    hannWindow = sig.hann(blockSize, True)

    spectralFlux = []
    block1 = np.zeros(blockSize)
    block1 *= hannWindow
    block1FFT = np.abs(np.fft.fft(block1)[:blockSize//2])
    
    for block2 in xb:    
        block2 *= hannWindow
        block2FFT = np.abs(np.fft.fft(block2)[:blockSize//2])
        flux = np.sqrt(np.sum((block2FFT - block1FFT)**2)) / (blockSize//2)
        spectralFlux.append(flux)
        block1FFT = block2FFT.copy()

    return np.array(spectralFlux)



def extract_features(x, blockSize, hopSize, fs):
    blocks, times = block_audio(x, blockSize, hopSize, fs)
    centroids = extract_spectral_centroid(blocks, fs)
    rms = extract_rms(blocks)
    zeroCrossings = extract_zerocrossingrate(blocks)
    crest = extract_spectral_crest(blocks)
    flux = extract_spectral_flux(blocks)
    return np.stack((centroids, rms, zeroCrossings, crest, flux))
     


def aggregate_feature_per_file(features):
    numFeatures = features.shape[0]
    aggFeatures = np.zeros(features.shape[0]*2)
    for i in range(numFeatures):
        aggFeatures[i*2] = np.mean(features[i])
        aggFeatures[i*2+1] = np.std(features[i])
    return aggFeatures.reshape(-1,1)



def get_feature_data(path, blockSize, hopSize):
    featureData = []
    for audioFile in glob.glob(os.path.join(path, '*')):
        fs, x = wavread(audioFile)
        features = extract_features(x, 1024, 256, fs)
        aggFeatures = aggregate_feature_per_file(features)
        featureData.append(aggFeatures)
    return np.array(featureData).squeeze().T



def normalize_zscore(featureData):
    return np.array([
        [(featureData[i][j]-mu)/std for j in range(featureData.shape[1])] for i,(mu,std) in enumerate([(np.mean(feature), np.std(feature)) for feature in featureData])
    ])



def visualize_features(path_to_musicspeech):

    speechFolder = glob.glob(os.path.join(path_to_musicspeech, 'speech_wav'))[0]
    musicFolder = glob.glob(os.path.join(path_to_musicspeech, 'music_wav'))[0]

    # Calculate Features
    speechFeatures = get_feature_data(speechFolder, 1024, 256)
    musicFeatures = get_feature_data(musicFolder, 1024, 256)
    combinedFeatures = np.concatenate((speechFeatures, musicFeatures), axis=1)

    # Z-score Normalization
    combinedFeatures = normalize_zscore(combinedFeatures)
    speechFeatures, musicFeatures = np.split(combinedFeatures, 2, axis=1)
    
    # Feature Visualization

    # SC Mean, SCR Mean
    plt.scatter(speechFeatures[0], speechFeatures[6], color='blue', label='Speech')
    plt.scatter(musicFeatures[0], musicFeatures[6], color='red', label='Music')
    plt.xlabel("Spectral Centroid Mean")
    plt.ylabel("Spectral Crest Mean")
    plt.grid()
    plt.legend()
    plt.show()

    # SF mean, ZCR mean
    plt.scatter(speechFeatures[8], speechFeatures[4], color='blue', label='Speech')
    plt.scatter(musicFeatures[8], musicFeatures[4], color='red', label='Music')
    plt.xlabel("Spectral Flux Mean")
    plt.ylabel("Zero Crossing Rate Mean")
    plt.grid()
    plt.legend()
    plt.show()

    # RMS mean, RMS std
    plt.scatter(speechFeatures[2], speechFeatures[3], color='blue', label='Speech')
    plt.scatter(musicFeatures[2], musicFeatures[3], color='red', label='Music')
    plt.xlabel("RMS Mean")
    plt.ylabel("RMS STD")
    plt.grid()
    plt.legend()
    plt.show()
    
    # ZCR std, SCR std
    plt.scatter(speechFeatures[5], speechFeatures[7], color='blue', label='Speech')
    plt.scatter(musicFeatures[5], musicFeatures[7], color='red', label='Music')
    plt.xlabel("Zero Crossing Rate STD")
    plt.ylabel("Spectral Crest STD")
    plt.grid()
    plt.legend()
    plt.show()
    
    # SC std, SF std
    plt.scatter(speechFeatures[1], speechFeatures[9], color='blue', label='Speech')
    plt.scatter(musicFeatures[1], musicFeatures[9], color='red', label='Music')
    plt.xlabel("Spectral Centroid STD")
    plt.ylabel("Spectral Flux STD")
    plt.grid()
    plt.legend()
    plt.show()

    
