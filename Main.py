from NNMIDI.InputController import MidiInput
from NNMIDI.ModelBulder import Model
from NNMIDI.OutputController import MidiOutput

input_data = MidiInput.build_data_set(64, 1, 'sonata.mid', 32)[0]
output_data = MidiOutput('midifile', 8, 4, 4)

model = Model.build_model(input_data, output_data, 64)
for i in range(10):
    model.train(1)
    (seed, generated) = model.generate(128, input_data.get_segment_data_set(0, 64))
    model.output_controller.build_midi_file("r" + str(i))
    model.output_controller = MidiOutput('midifile', 8, 4, 4)

#
model.save('exp')
