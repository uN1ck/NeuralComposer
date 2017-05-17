from CustomMidi.Orchestra import build_orchestra

orchestra = build_orchestra(input_path="train_ez", output_path="res", division_in=32, division_out=32, sample_length=8)
for i in range(20):
    orchestra.train(10)
