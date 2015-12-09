import numpy
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import RecurrentNetwork
from pybrain.tools.customxml import NetworkWriter, NetworkReader
from pybrain.datasets import SequenceClassificationDataSet
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.structure.modules import LSTMLayer, SigmoidLayer, LinearLayer, SoftmaxLayer
from pybrain.structure.moduleslice import ModuleSlice
from pybrain.structure import FullConnection, MotherConnection, SharedFullConnection
import pickle
import sys
import os
import glob
import build_data
import wave_gen
import wave_reader
import midi_util


def saveFile(filename, data):
	numpy.savetxt(filename, data, delimiter=',', fmt='%.2f', newline = '\n')

def saveNetwork(filename, net):
	fileObject = open(filename, 'w')
	pickle.dump(net, fileObject)
	fileObject.close()

def loadNetwork(filename):
	fileObject = open(filename, 'r')
	return pickle.load(fileObject)

def trainNetwork(dirname):

    numFeatures = 2025

    ds = SequenceClassificationDataSet(numFeatures, 1, nb_classes=60)
    
    tracks = glob.glob(os.path.join(dirname, '*.csv'))
    for t in tracks:
        track = os.path.splitext(t)[0]
        # load training data
        print "Reading %s..." % t
        data = numpy.genfromtxt(t, delimiter=",")
        numData = data.shape[0]

        # add the input to the dataset
        print "Adding to dataset..."
        ds.newSequence()
        for i in range(numData):
            #ds.addSample(data[i], (labels[i],))
            input = data[i]
            label = input[numFeatures]
            if label > 0:
                label = max(0,midi_util.frequencyToMidi(label)-36) #36 is our lowest MIDI note we consider
            ds.addSample(input[0:numFeatures],(label,))
    ds._convertToOneOfMany(bounds=[0,1]) #change label to indicator vector

    # initialize the neural network
    print "Initializing neural network..."
    #manual network building
    net = RecurrentNetwork()
    inlayer = LinearLayer(numFeatures)
    hiddenLayer = LSTMLayer(17)
    #hiddenLayer2 = SigmoidLayer(17)
    outlayer = SoftmaxLayer(60)

    net.addInputModule(inlayer)
    net.addOutputModule(outlayer)
    net.addModule(hiddenLayer)
    #net.addModule(hiddenLayer2)

    net.addConnection(FullConnection(inlayer, hiddenLayer, inSliceFrom=2000, outSliceTo=5))
    net.addConnection(FullConnection(inlayer, hiddenLayer, outSliceFrom=5))
    #net.addConnection(FullConnection(hiddenLayer, hiddenLayer2, inSliceTo=5, outSliceTo=5))
    #net.addConnection(FullConnection(hiddenLayer, hiddenLayer2, inSliceFrom=5, outSliceFrom=5))
    for i in range(5):
        net.addConnection(FullConnection(hiddenLayer, outlayer, inSliceFrom=i, inSliceTo=i+1, outSliceFrom=i*12, outSliceTo=(i+1)*12))

        
    net.addConnection(FullConnection(hiddenLayer, outlayer, inSliceFrom=5))

    net.sortModules()


    # train the network on the dataset
    print "Training neural net"
    trainer = RPropMinusTrainer(net, dataset=ds)
    #error = trainer.trainUntilConvergence(maxEpochs=700, verbose=True, continueEpochs=30, validationProportion=0.2)
    error = -1
    errors = []
    for i in range(1000):
        new_error = trainer.train()
        print "error: " + str(new_error)
        errors.append(new_error)
        if abs(error - new_error) < 0.00000002: break
        error = new_error
    # save the network
    print "Saving neural network..."
    NetworkWriter.writeToFile(net, '1hiddenlayernet')
    print "Error:"
    print errors
    saveFile(os.path.join(dirname,'errors'), errors)
    

if __name__ == '__main__':
    dirname = os.path.normpath(sys.argv[1])
    # wave_reader.extractFeatures(track)
    trainNetwork(dirname)
    #use tester.py if interested in trying out a prediction with the network
