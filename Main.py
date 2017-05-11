from NNMIDI.InputController import MidiInput
from NNMIDI.ModelBulder import Model
from NNMIDI.OutputController import MidiOutput

input_data = MidiInput(64, 1, 'sonata.mid', 32)[0]
output_data = MidiOutput.build_data_out('midifile', 64, 4, 4, 120, 2)  # ('log', 'midi_data')

model = Model.build_model(input_data, output_data, 64)
for i in range(10):
    model.train(1)
    (seed, generated) = model.generate(128, input_data.get_segment_data_set(0, 64))
    model.output_controller.build_midi_file()
    model.output_controller.midi.save('b-' + str(i) + '.mid')

output_data.close()
#
model.save('exp')
