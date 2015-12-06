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

def wavToFeatures(filename):
    rate, data = wavfile.read(filename)
    if data.ndim > 1:
        data = stereoToMono(data)

    segDiv = 4 #1/10 of a second
    segSize = rate / segDiv
    start = 0
    end = segSize
    matrix = numpy.array([])
    for i in range(0, data.size / segSize):
        data_seg = data[start:end]
        start += segSize
        end += segSize
        features = analyzer.getFrequencies(data_seg)
        matrix = build_data.addToFeatureMatrix(matrix, features[0:2000])
    # build_data.saveFile('testdata.csv', matrix)
    return matrix

def extractFeatures(track, div):
    rate, data = wavfile.read(track + '.wav')
    if data.ndim > 1:
        data = stereoToMono(data)
    matrix = numpy.array([])
    vmatrix = numpy.array([])

    segSize = rate / div
    start = 0
    end = segSize
    for i in range(0, data.size / segSize):
        data_seg = data[start:end]
        start += segSize
        end += segSize
        features = analyzer.getFrequencies(data_seg)[0:5000]
        matrix = build_data.addToFeatureMatrix(matrix, features)
        vmatrix = build_data.addToFeatureMatrix(vmatrix,
                             numpy.array([numpy.argmax(features)]))
    build_data.saveFile(track + '_seg.csv', matrix)
    build_data.saveFile(track + '_volume.csv', vmatrix)

if __name__ == '__main__':
    filename = sys.argv[1]
    rate, data = wavfile.read(filename)
    if data.ndim > 1:
        data = stereoToMono(data)

    segDiv = 10
    segSize = rate / segDiv
    start = 0
    end = segSize
    matrix = numpy.array([])
    vmatrix = numpy.array([])
    for i in range(0, data.size / segSize):
        data_seg = data[start:end]
        start += segSize
        end += segSize
        features = analyzer.getFrequencies(data_seg)
        matrix = build_data.addToFeatureMatrix(matrix, features[0:5000])
        vmatrix = build_data.addToFeatureMatrix(vmatrix,
                             numpy.floor(numpy.round(features[0:5000], 5)))
    # build_data.saveFile('testdata.csv', matrix)
    build_data.saveFile('volumedata.csv', vmatrix)

    # wave_gen.writeWav('converted_' + filename, matrix, segDiv)
    wave_gen.writeWav('volume_single_' + filename, vmatrix, segDiv)
