import glob
import music21
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,Activation
from keras.utils import np_utils
from music21 import converter, instrument, note, chord
import numpy as np
from tensorflow.python.keras.models import load_model

notes = []

for file in glob.glob("C://Users//SHAASHWAT//Downloads//Classical-Piano-Composer-master//midi_songs//*.mid"):
    midi = converter.parse(file)
    #print(midi)
    notes_to_parse=None


    parts=instrument.partitionByInstrument(midi)
    #print(parts)

    if parts:  # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

#print(notes)

sequence_length = 100

pitch_names=sorted((set(item for item in notes)))
note_to_int = dict((note, number) for number,note in enumerate(pitch_names))

network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i+sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])


n_patterns = len(network_input)
n_vocab = 358
network_input = np.reshape(network_input, (n_patterns, sequence_length,1))
network_input = network_input / float(n_vocab)
network_output = np_utils.to_categorical(network_output)



#THE MODEL

model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


filepath = "weights-improvement-{epoch:02(d+30)}-{loss:.4f}-bigger.hdf5"

checkpoint = ModelCheckpoint(
    filepath, monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)
callbacks_list = [checkpoint]
model=load_model('weights-improvement-30-1.9522-bigger.hdf5')
model.fit(network_input, network_output, epochs=170, batch_size=64,callbacks=callbacks_list)


#print(network_output)