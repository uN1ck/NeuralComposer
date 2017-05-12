from CustomMidi.CustomMidiFile import CustomMidiFile, build_custom_midi_file
from CustomMidi.CustomTrack import CustomTrack
from NNMIDI.ModelBulder import build_model

input_data = build_custom_midi_file('sonata.mid', 16)

model = build_model(input_data, 64)
for i in range(10):
    model.train(1)
    (seed, generated) = model.generate(128, input_data.tracks[0].get_segment_data_set(0, 64))

    output_data = CustomMidiFile()
    output_track = CustomTrack(16)
    output_track.divisions = generated
    output_data.tracks.append(output_track)
    output_data.build_midi_file("r" + str(i), 120, 4, 4)
#
model.save('exp')
