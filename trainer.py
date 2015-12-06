import numpy
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.customxml import NetworkWriter, NetworkReader
from pybrain.datasets import SequentialDataSet
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.structure.modules import LSTMLayer
import pickle
import sys
import os
import glob
import build_data
import wave_gen
import wave_reader

def saveNetwork(filename, net):
	fileObject = open(filename, 'w')
	pickle.dump(net, fileObject)
	fileObject.close()

def loadNetwork(filename):
	fileObject = open(filename, 'r')
	return pickle.load(fileObject)

def trainNetwork(dirname):
    numFeatures = 5000
    ds = SequentialDataSet(numFeatures, 1)
    
    tracks = glob.glob(os.path.join(dirname, '*_.wav'))
    #melodies = glob.glob(os.path.join(dirname, '*_melody.wav'))
    for t in tracks:
        track = os.path.splitext(t)[0]
        # load training data
        print "Reading %s..." % track
        data = wave_reader.wavToFeatures("%s.wav" %track)
        melody = wave_reader.wavToFeatures("%smelody.wav" %track)
        labels = melody.argmax(1) #axis=1, max frequency across a sample
        numData = data.shape[0]

        # add the input to the dataset
        print "Adding to dataset..."
        ds.newSequence()
        for i in range(numData):
            ds.addSample(data[i], (labels[i],))

    # save the dataset
    print "Saving dataset..."
    ds.saveToFile(os.path.basename(dirname) + 'constructedDataset')



    # initialize the neural network
    print "Initializing neural network..."
    net = buildNetwork(numFeatures, 50, 1,
                       hiddenclass=LSTMLayer, outputbias=False, recurrent=True)
    
    # train the network on the dataset
    print "Training neural net"
    trainer = RPropMinusTrainer(net, dataset=ds)
##    trainer.trainUntilConvergence(maxEpochs=50, verbose=True, validationProportion=0.1)
    error = -1
    for i in range(50):
        new_error = trainer.train()
        print "error: " + str(new_error)
        if abs(error - new_error) < 0.1: break
        error = new_error

    # save the network
    print "Saving neural network..."
    NetworkWriter.writeToFile(net, os.path.basename(dirname) + 'net')

if __name__ == '__main__':
    dirname = os.path.normpath(sys.argv[1])
    # wave_reader.extractFeatures(track)
    trainNetwork(dirname)

    '''
    net = NetworkReader.readFrom(os.path.basename(dirname) + 'net')
    
    # predict on some of the training examples
    print "Predicting on training set"
    data = numpy.genfromtxt(os.path.join(dirname, 'train09_seg.csv'), delimiter=",")
    labels = numpy.genfromtxt(os.path.join(dirname, 'train09REF.txt'), delimiter='\t')[:,1]
    for i in range(200):
        print net.activate(data[i]), labels[i]
##    cdata = numpy.array([])
##    for feature in data:
##        freq = max(0, net.activate(feature))
##        sample = wave_gen.saw(freq, 0.1, 44100)
##        cdata = numpy.concatenate([cdata, sample])
##    wave_gen.saveAudioBuffer('test.wav', cdata)
##    for freq in labels:
##        sample = wave_gen.saw(freq, 0.05, 44100)
##        cdata = numpy.concatenate([cdata, sample])
##    wave_gen.saveAudioBuffer('test_ref.wav', cdata)'''
