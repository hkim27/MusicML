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

    numFeatures = 2000

    ds = SequentialDataSet(numFeatures, 1)
    
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
            ds.addSample(input[0:numFeatures],(input[numFeatures],))


    # initialize the neural network
    print "Initializing neural network..."
    net = buildNetwork(numFeatures, 50, 1,
                       hiddenclass=LSTMLayer, bias=True, recurrent=True)
    
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
    NetworkWriter.writeToFile(net, os.path.basename(dirname) + 'net' + os.getenv('HOSTNAME'))

if __name__ == '__main__':
    dirname = os.path.normpath(sys.argv[1])
    # wave_reader.extractFeatures(track)
    trainNetwork(dirname)
    #use tester.py if interested in trying out a prediction with the network
