import math
import numpy
import scipy.io.wavfile as wavfile
from scipy import signal
import sys

def sine(frequency, length, rate):
    length = int(length * rate)
    factor = float(frequency) * (math.pi * 2) / rate
    return numpy.sin(numpy.arange(length) * factor)

def square(frequency, length, rate):
    length = int(length * rate)
    factor = float(frequency) * (math.pi * 2) / rate
    return signal.square(numpy.arange(length) * factor)
    # t = numpy.linspace(0, length, rate, endpoint=False)
    # return signal.square(2 * numpy.pi * 5 * t)

def saw(frequency, length, rate):
    length = int(length * rate)
    factor = float(frequency) * (math.pi * 2) / rate
    return signal.sawtooth(numpy.arange(length) * factor)
    # t = numpy.linspace(0, length, rate, endpoint=False)
    # return signal.sawtooth(2 * numpy.pi * 5 * t)

def noise(length, rate):
    return numpy.random.uniform(-1, 1, length * rate)

def saveAudioBuffer(filename, data):
    scaled = numpy.int16(data/float(numpy.max(numpy.abs(data))) * 32767)
    wavfile.write(filename, 44100, scaled)

def generateAudio(frequency=440, length=1, rate=44100):
    chunks = []
    chunks.append(sine(frequency, length, rate))

    chunk = numpy.concatenate(chunks) * 0.25

def writeWav(filename, cmatrix, div):
    cdata = numpy.array([])
    for i, feature in enumerate(cmatrix):
        sys.stdout.write(str(i * 100/len(cmatrix)) + '%\r')
        sample = numpy.zeros(44100 / div)
        for freq in range(5000):
            if feature[freq] > 0:
                sample += feature[freq] * saw(freq, 1.0 / div, 44100)
        cdata = numpy.concatenate([cdata, sample])
    saveAudioBuffer(filename, cdata)

if __name__ == '__main__':
    data = saw(116, 1, 44100)
    scaled = numpy.int16(data/numpy.max(numpy.abs(data)) * 32767)
    wavfile.write('test.wav', 44100, scaled)

