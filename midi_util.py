import math

def midiToFrequency(note):
	return (math.pow(2, ((note - 69) / 12.0)) * 440)

def frequencyToMidi(freq):
	return 69 + round(12*math.log(freq/440.0,2.0))

def frequencyToNoteFrequency(freq):
	return midiToFrequency(frequencyToMidi(freq))