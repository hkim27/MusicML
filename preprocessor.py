import numpy
import sys
import os
import glob
import midi
import midi_util
import wave_gen
import sequencer

#Run this to convert midi files (chorales) in the specified directory into wav + a wav of the melody line
#The resulting files from any *.mid will be *.wav and *melody.wav in the same directory
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
