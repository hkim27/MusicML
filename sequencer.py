import numpy
import midi
import midi_util
import wave_gen

# midi has 7 bits (128 different notes)
class Sequencer:
	def __init__(self):
		# create the empty note event arrays
		self.noteActive = [False] * 128
		self.sampleRate = 44100
		self.output = [numpy.empty(0),numpy.empty(0)]
		self.tracks = []
		self.bpm = 120 #default tempo
		self.ppq = 96 #default resolution

	def generateAudio(self, duration, track):
		if duration == 0:
			return

		# generate the base empty sample
		seconds = (60.0 / (self.bpm * self.ppq)) * duration
		length = int(seconds * self.sampleRate)
		sample = numpy.zeros(length)
		melodysample = numpy.zeros(length)
		# add in any notes that are active
		#print "adding some samples in"
		for i in range(0, 128):
			if (self.noteActive[i]):
				#print "note: " + str(i)
				frequency = midi_util.midiToFrequency(i)
				#print "adding frequency " + str(frequency)
				sample += wave_gen.saw(frequency, seconds, self.sampleRate)
				if track == 0: #melody track for chorales
					melodysample += wave_gen.sine(frequency, seconds, self.sampleRate)

		# concatenate this to the output
		self.tracks[track] = numpy.concatenate([self.tracks[track], sample])
		if track == 0: #melody track for chorales
			self.output[1] = numpy.concatenate([self.output[1], melodysample])



	def parseMidiFile(self, filename):
		#self.output = [] #numpy.zeros(5)
		pattern = midi.read_midifile(filename)
		self.ppq = pattern.resolution
		for track in pattern:
			#print track
			self.tracks.append([]) #add a new empty track
			for event in track:
				#print repr(event)

				if type(event) is midi.events.SetTempoEvent:
					self.bpm = event.get_bpm() #change tempo

				if type(event) is midi.events.NoteOnEvent:
					self.generateAudio(event.tick, event.channel)
					note = event.pitch
					self.noteActive[note] = True
				elif type(event) is midi.events.NoteOffEvent:
					self.generateAudio(event.tick, event.channel)
					note = event.pitch
					self.noteActive[note] = False
		self.tracks.pop() #removes the extra added track (the first track is just metadata)

		track_len = min([len(track) for track in self.tracks])
		self.tracks = [track[0:track_len] for track in self.tracks] #trim all tracks to the exact same size
		self.output[0] = numpy.sum(self.tracks, axis=0) #combine tracks
		return self.output

	def resetSequencer(self):
		self.noteActive = [False] * 128
		self.sampleRate = 44100
		self.output = [numpy.empty(0),numpy.empty(0)]
		self.tracks = []
		self.bpm = 120 #default tempo
		self.ppq = 96 #default resolution

if __name__ == '__main__':
	seq = Sequencer()
	samples = seq.parseMidiFile('000106b_.mid')
	wave_gen.saveAudioBuffer('000106b_.wav', samples)
	#samples = seq.parseMidiFile('mary.mid')
	#wave_gen.saveAudioBuffer('mary.wav', samples)
