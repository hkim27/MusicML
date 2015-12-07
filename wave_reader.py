import math
import numpy
import scipy.io.wavfile as wavfile
from scipy import signal
import analyzer
import build_data
import sys
import wave_gen

def stereoToMono(audiodata):
    d = audiodata.sum(axis=1) / 2
    return d

def readWav(filename):
    rate, data = wavfile.read(filename)
    return rate, data

def wavToFeatures(track):
    rate, data = wavfile.read(track)
    if data.ndim > 1:
        data = stereoToMono(data)
    matrix = numpy.array([])

    segSize = rate / 4
    start = 0
    end = segSize
    for i in range(0, data.size / segSize):
        data_seg = data[start:end]
        start += segSize
        end += segSize
        features = analyzer.getFrequencies(data_seg)[0:2000]
        matrix = build_data.addToFeatureMatrix(matrix, features)
    return matrix

if __name__ == '__main__':
    track = sys.argv[1]
    wavToFeatures(track)
##    matrix = numpy.genfromtxt(track + '_seg.csv', delimiter=',')
##    wave_gen.writeWav(track + '_seg.wav', matrix, div)
##    vmatrix = numpy.genfromtxt(track + '_volume.csv', delimiter=',')
##    wave_gen.writeWav(track + '_volume.wav', vmatrix, div)
