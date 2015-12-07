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
    #net = buildNetwork(numFeatures, 50, 1,
    #                   hiddenclass=LSTMLayer, bias=True, recurrent=True)

    #manual network building
    net = RecurrentNetwork()
    inlayer = LinearLayer(numFeatures)
    #h1 = LSTMLayer(70)
    #h2 = SigmoidLayer(50)
    octaveLayer = LSTMLayer(5)
    noteLayer = LSTMLayer(12)
    outlayer = SoftmaxLayer(60)
    #outlayer = LinearLayer(1)

    net.addInputModule(inlayer)
    net.addOutputModule(outlayer)
    #net.addModule(h1)
    #net.addModule(h2)
    #net.addModule(hiddenLayer)
    net.addModule(octaveLayer)
    net.addModule(noteLayer)
    #net.addModule(combinedLayer)
    
    #net.addConnection(FullConnection(inlayer, h1))
    #net.addConnection(FullConnection(h1, h2))
    #net.addConnection(FullConnection(h2, outlayer))
    net.addConnection(FullConnection(inlayer, octaveLayer, inSliceFrom=2000))
    net.addConnection(FullConnection(inlayer, noteLayer))
    #net.addConnection(FullConnection(octaveLayer,outlayer))
    motherConnection = []
    for i in range(5):
        motherConnection.append(MotherConnection(12, name="%d"%i))
        net.addConnection(SharedFullConnection(motherConnection[i], octaveLayer, outlayer, inSliceFrom=i, inSliceTo=i+1, outSliceFrom=i*12, outSliceTo=(i+1)*12))
        
    net.addConnection(FullConnection(noteLayer,outlayer))
    #net.addConnection(FullConnection(combinedLayer, outlayer))

    net.sortModules()


    # train the network on the dataset
    print "Training neural net"
    trainer = RPropMinusTrainer(net, dataset=ds)
##    trainer.trainUntilConvergence(maxEpochs=50, verbose=True, validationProportion=0.1)
    error = -1
    for i in range(250):
        new_error = trainer.train()
        print "error: " + str(new_error)
        if abs(error - new_error) < 0.0000002: break
        error = new_error

    # save the network
    print "Saving neural network..."
    NetworkWriter.writeToFile(net, os.path.basename(dirname) + 'classifSharednet2')

if __name__ == '__main__':
    dirname = os.path.normpath(sys.argv[1])
    # wave_reader.extractFeatures(track)
    trainNetwork(dirname)
    #use tester.py if interested in trying out a prediction with the network
