import numpy
from pybrain.tools.customxml import NetworkWriter, NetworkReader
import sys
import os
import wave_gen
import wave_reader
import midi_util
import glob

# Script to test a trained neural net on a wav file (000106b_.wav)
#produces a result wav so you can listen to the prediction
if __name__ == '__main__':
    dirname = os.path.normpath(sys.argv[1])

    print "Reloading neural network..."
    net = NetworkReader.readFrom(os.path.basename(dirname) + 'designsmaxnet')
    
    tracks = glob.glob(os.path.join(dirname, 'testing/*.csv'))
    tracks = tracks[3:5]
    for t in tracks:
        track = os.path.splitext(t)[0]
        print "Reading processed %s..." %track
        data = numpy.genfromtxt(track + '.csv', delimiter=",")
        numData = data.shape[0]
        numFeatures = data.shape[1]-1
        features = data[:,0:numFeatures]
        labels = data[:,numFeatures]
        predict = numpy.zeros(numData)

        print "Activating neural network..."
        for i in range(numData):
            indicatorVector = numpy.array(net.activate(features[i]))
            predict[i] = indicatorVector.argmax() + 36 #lowest MIDI in range is 36
            
        print "Generating result %s.wav..." %track
        cdata = numpy.array([])
        for pred in predict:
            #if(freq > 0):
            #    freq = midi_util.frequencyToNoteFrequency(label)
            pred = round(pred)
            freq = midi_util.midiToFrequency(pred)
            sample = wave_gen.saw(freq, 0.25, 44100)
            cdata = numpy.concatenate([cdata, sample])

        print "Saving result %s.wav..." %track
        wave_gen.saveAudioBuffer('%sresult.wav' %track, cdata)
