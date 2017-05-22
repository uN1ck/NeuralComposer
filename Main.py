from CustomMidi.Musician import threshold_sequence_max_delta
from CustomMidi.Orchestra import build_orchestra

orchestra = build_orchestra(division_in=32,
                            division_out=32,
                            sample_length=16,
                            output_length=8,
                            thresholder=threshold_sequence_max_delta,
                            loss='mean_squared_error',
                            optimizer='RMSprop')
for i in range(20):
    orchestra.train(5)
