from CustomMidi.Musician import threshold_sequence_max_delta, threshold_sequence_max
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

orchestra = build_orchestra(division_in=32,
                            division_out=32,
                            sample_length=16,
                            output_length=4,
                            thresholder=threshold_sequence_max_delta,
                            loss='mean_absolute_error',
                            optimizer='RMSprop')
for i in range(20):
    orchestra.train(5)

orchestra = build_orchestra(division_in=32,
                            division_out=32,
                            sample_length=16,
                            output_length=4,
                            thresholder=threshold_sequence_max,
                            loss='mean_absolute_error',
                            optimizer='RMSprop')
for i in range(20):
    orchestra.train(5)

orchestra = build_orchestra(division_in=32,
                            division_out=32,
                            sample_length=16,
                            output_length=8,
                            thresholder=threshold_sequence_max_delta,
                            loss='mean_squared_error',
                            optimizer='Adam')
for i in range(20):
    orchestra.train(5)

orchestra = build_orchestra(division_in=32,
                            division_out=32,
                            sample_length=16,
                            output_length=4,
                            thresholder=threshold_sequence_max_delta,
                            loss='mean_absolute_error',
                            optimizer='Adam')
for i in range(20):
    orchestra.train(5)

orchestra = build_orchestra(division_in=32,
                            division_out=32,
                            sample_length=16,
                            output_length=4,
                            thresholder=threshold_sequence_max,
                            loss='mean_absolute_error',
                            optimizer='Adam')
for i in range(20):
    orchestra.train(5)

orchestra = build_orchestra(division_in=32,
                            division_out=32,
                            sample_length=16,
                            output_length=8,
                            thresholder=threshold_sequence_max_delta,
                            loss='mean_squared_error',
                            optimizer='Nadam')
for i in range(20):
    orchestra.train(5)

orchestra = build_orchestra(division_in=32,
                            division_out=32,
                            sample_length=16,
                            output_length=4,
                            thresholder=threshold_sequence_max_delta,
                            loss='mean_absolute_error',
                            optimizer='Nadam')
for i in range(20):
    orchestra.train(5)

orchestra = build_orchestra(division_in=32,
                            division_out=32,
                            sample_length=16,
                            output_length=4,
                            thresholder=threshold_sequence_max,
                            loss='mean_absolute_error',
                            optimizer='Nadam')
for i in range(20):
    orchestra.train(5)
