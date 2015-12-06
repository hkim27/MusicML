import numpy
import sys
import os
import glob
import midi
import midi_util
import wave_gen
import sequencer

if __name__ == '__main__':
    dirname = os.path.normpath(sys.argv[1])
    seq = sequencer.Sequencer() #create a sequencer to handle midi input


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
        seq.resetSequencer()
        # add the input to the dataset
        #print "Adding to dataset..."


#	samples = seq.parseMidiFile('000106b_.mid')
#	wave_gen.saveAudioBuffer('000106b_.wav', samples)