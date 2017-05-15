from CustomMidi.Orchestra import build_orchestra

orchestra = build_orchestra("2.mid", "result.mid", 32, 32, 4)
for i in range(20):
    orchestra.train(10, 25)
    orchestra.output.build_midi_file("r" + str(i), 4, 4)
