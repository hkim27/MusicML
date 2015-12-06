import math

def midiToFrequency(note):
	return (math.pow(2, ((note - 69) / 12.0)) * 440)

def frequencyToMidi(freq):
	return max(0,69 + round(12*math.log(freq/440.0,2.0))) #prevent negative midi numbers - 0 is C 5 octaves below middle C

def frequencyToNoteFrequency(freq):
	return midiToFrequency(frequencyToMidi(freq))