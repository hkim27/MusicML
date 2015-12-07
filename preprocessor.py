import numpy
import sys
import os
import glob
import midi
import midi_util
import wave_gen
import wave_reader
import sequencer
import svm_predict


def saveFile(filename, data):
	numpy.savetxt(filename, data, delimiter=',', fmt='%.2f', newline = '\n')

def chordArray(basenote):
    array = numpy.zeros(24)
    array[basenote] = 1
    return array


#Run this to convert midi files (chorales) in the specified directory into wav + a wav of the melody line
#The resulting files from any *.mid will be *.wav and *melody.wav in the same directory
if __name__ == '__main__':
    dirname = os.path.normpath(sys.argv[1])
    seq = sequencer.Sequencer() #create a sequencer to handle midi input
    chordsvm = svm_predict.loadSVM()


    tracks = glob.glob(os.path.join(dirname, '*.mid')) #match any midi file
    for t in tracks:
        track = os.path.splitext(t)[0]
        # sequence MIDI to saw + sine melody
        print "Sequencing %s..." %track
        wavData = seq.parseMidiFile("%s.mid" %track)
        sawData = wavData[0]
        melodyData = wavData[1]


        print "Saving %s... wav and melody" %track
        wave_gen.saveAudioBuffer("%s.wav" %track, sawData)
        wave_gen.saveAudioBuffer("%smelody.wav" %track, melodyData)

        print "Generating feature data..."
        melody = wave_reader.wavToFeatures("%smelody.wav" %track)
        labels = numpy.array([melody.argmax(axis=1)]) #axis=1, max frequency across a sample
        data = wave_reader.wavToFeatures("%s.wav" %track)
        numData = min(data.shape[0],labels.shape[0])

        labels = labels[0:numData]
        data = data[0:numData]
        chord = numpy.array([chordArray(chordsvm.predict(feature)) for feature in data])
        volume = numpy.array([data.argmax(axis=1)])

        matrix = numpy.concatenate([data, chord, volume.T, labels.T], axis=1) #horizontal concatenate
        saveFile('%sdata.csv' %track, matrix)

        seq.resetSequencer()

