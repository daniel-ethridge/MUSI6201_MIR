import numpy as np
import scipy
from scipy.ndimage.interpolation import shift
import math
import matplotlib.pyplot as plt
import pss_madmom as pss

dirname = '/Users/ethri/Desktop/MIR/MIR_Project/MDBDrums/MDB Drums/audio/drum_only'

def  block_audio(x,blockSize,hopSize,fs):    
    # allocate memory    
    numBlocks = math.ceil(x.size / hopSize)   
    xb = np.zeros([numBlocks, blockSize])    
    # compute time stamps    
    t = (np.arange(0, numBlocks) * hopSize) / fs    
    
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)  
    
    for n in range(0, numBlocks):        
        i_start = n * hopSize        
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])        
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]

    return xb, t, numBlocks

def DC_filter(x, blockSize, hopSize, fs):
    xb, t, numblocks = block_audio(x,blockSize,hopSize,fs)
    for r in range(1, xb.shape[0]):
        for c in range(1, xb.shape[-1]):
            xb[r,c] = xb[r,c] - xb[r].mean()
    
    return xb        

def compute_hann(window_length):
    window_array = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(window_length) / window_length)
    return window_array

#def temporal_centroid(xb):
#    block_size = xb.shape[-1]
#    num_of_block = xb.shape[0]
#    window = compute_hann(block_size)
#    Xb = np.zeros(xb.shape)
#    for n in range(0, num_of_block):
#        Xb[n] = xb[n] * window
        
def spectral_kurtosis(xb):
    blockSize = xb.shape[-1]
    num_of_blocks = xb.shape[0]
    window = compute_hann(block_Size)
    Xb = np.zeros(xb.shape)
    for n in range(0, num_of_blocks):
        Xb[n] = xb[n] * window
        Xb[n] = np.absolute(np.fft.fft(Xb[n]))
    spectralKurtosis = np.zeros(num_of_blocks)
    for n in range(0, num_of_blocks):
        for k in range(0, (blockSize//2)-1):
            spectralKurtosis[n,k] = -3 + sum(np.power(Xb[n,k] - Xb[n].mean(), 4)[0:(blockSize/2)-1]) // (blockSize * np.power(Xb[n].std, 4))
    
                                           
def extract_spectral_centroid(xb, fs):
    block_size = xb.shape[-1]
    num_of_block = xb.shape[0]
    window = compute_hann(block_size)
    Xb = np.zeros(xb.shape)
    for n in range(0, num_of_block):
        Xb[n] = xb[n] * window
        Xb[n] = np.absolute(np.fft.fft(Xb[n]))
    vsc_freq_bin = np.zeros(num_of_block)
    for i in range(0, num_of_block):
        vsc_freq_bin[i] = np.sum(np.multiply(np.array(
            range(0, block_size//2)),
            Xb[i, 0:block_size//2])) / \
                          np.sum(Xb[i, 0:block_size//2])
    vsc_freq_hz = vsc_freq_bin * fs / block_size
    return vsc_freq_hz


def extract_rms(xb):
    block_size = xb.shape[-1]
    num_of_block = xb.shape[0]
    rms_linear = np.zeros(num_of_block)
    for i in range(0, num_of_block):
        rms_linear[i] = math.sqrt(np.sum(np.square(xb[i, :], xb[i, :])) / block_size)
    rms_dB = 20 * np.log(rms_linear)
    rms_dB[rms_dB < -100] = -100
    return rms_dB


def extract_zerocrossingrate(xb):
    block_size = xb.shape[-1]
    num_of_block = xb.shape[0]
    zero_crossing_rate = np.zeros(num_of_block)
    for i in range(0, num_of_block):
        x_sign = xb[i]
        x_sign[x_sign < 0] = -1
        x_sign[x_sign > 0] = 1
        x_sign[x_sign == 0] = 0
        x_shift = shift(x_sign, 1, cval=0)
        zero_crossing_rate[i] = np.sum(x_sign - x_shift) / (2 * block_size)

    return zero_crossing_rate


def extract_spectral_crest(xb):
    block_size = xb.shape[-1]
    num_of_block = xb.shape[0]
    spectral_crest = np.zeros(num_of_block)
    window = compute_hann(block_size)
    Xb = np.zeros(xb.shape)
    for n in range(0, num_of_block):
        Xb[n] = xb[n] * window
        Xb[n] = np.absolute(np.fft.fft(Xb[n]))
    for i in range(0, num_of_block):
        spectral_crest[i] = Xb[i, 0:block_size // 2].max() / np.sum(Xb[i, 0:block_size // 2])
    return spectral_crest


def extract_spectral_flux(xb):
    block_size = xb.shape[-1]
    num_of_block = xb.shape[0]
    spectral_flux = np.zeros(num_of_block)
    window = compute_hann(block_size)
    Xb = np.zeros(xb.shape)
    for n in range(0, num_of_block):
        Xb[n] = xb[n] * window
        Xb[n] = np.absolute(np.fft.fft(Xb[n]))

    # spectral_flux[0] = math.sqrt(np.sum(np.square(Xb[0], Xb[0]))) / (block_size//2)
    for i in range(1, num_of_block):
        Xb_diff = Xb[i] - Xb[i - 1]
        spectral_flux[i] = math.sqrt(np.sum(np.square(Xb_diff, Xb_diff))) / (block_size // 2)

    return spectral_flux

def feature_extract():
    for i in range(0, len(pss.winvector)):
        if pss.winvector[i] == 1:
            pass
        
