import numpy
import cpickle as pickle
from sklearn import svm

def saveSVM(clf):
        fileObject = open('svm', 'w')
	pickle.dump(clf, fileObject)
	fileObject.close()

def loadSVM():
        fileObject = open('svm', 'r')
        return pickle.load(fileObject)

def trainSVM():
        # read the training file
        print "Reading in training file..."
        raw_data = numpy.genfromtxt('largedata.csv', delimiter=",")
        numFeatures = raw_data[0].size
        numData = raw_data.size / numFeatures

        X = raw_data[:, 0:numFeatures - 1]
        y = raw_data[:, numFeatures - 1]
        labels = numpy.copy(y)
        features = numpy.copy(X)
        clf = svm.SVC()
        print "Fitting the data"
        clf.fit(X, y)
        saveSVM(clf)

        print "Predicting on training data"
        predictions = clf.predict(features)
        errors = 0
        print "Predictions: "
        for i in range(numData):
                if predictions[i] != labels[i]:
                        errors += 1
                        print "guessed: " + str(predictions[i]) + ", actual: " + str(labels[i])
        print "error rate: " + str(errors * 1.0 / numData)

if __name__ == '__main__':
        # trainSVM()
        clf = loadSVM()
        testdata = numpy.genfromtxt('testdata.csv', delimiter=",")
        print clf.predict(testdata)
