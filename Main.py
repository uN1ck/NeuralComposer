from CustomMidi.Orchestra import build_orchestra

orchestra = build_orchestra(division_in=32, division_out=32, sample_length=16,
                            output_length=8)
for i in range(10):
    orchestra.train(1)
