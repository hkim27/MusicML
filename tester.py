import numpy
from pybrain.tools.customxml import NetworkWriter, NetworkReader
import sys
import os
import wave_gen
import wave_reader
import midi_util

# Script to test a trained neural net on a wav file (000106b_.wav)
#produces a result wav so you can listen to the prediction
if __name__ == '__main__':
    dirname = os.path.normpath(sys.argv[1])
    #track = os.path.join(dirname, '000106b_.wav')

    print "Reading wav into input data..."
    data = wave_reader.wavToFeatures('000106b_.wav')
    numData = data.shape[0]
    labels = numpy.zeros(numData)

    print "Reloading neural network..."
    net = NetworkReader.readFrom(os.path.basename(dirname) + 'designnet')

    print "Activating neural network..."
    for i in range(numData):
        labels[i] = net.activate(data[i])

    print "Generating result wav..."
    cdata = numpy.array([])
    for label in labels:
        #if(freq > 0):
        #    freq = midi_util.frequencyToNoteFrequency(label)
        label = round(label)
        freq = midi_util.midiToFrequency(label)
        sample = wave_gen.saw(freq, 0.25, 44100)
        sample = wave_gen.saw(freq, 0.25, 44100)
        cdata = numpy.concatenate([cdata, sample])

    print "Saving result wav..."
    wave_gen.saveAudioBuffer('000106b_predict.wav', cdata)
