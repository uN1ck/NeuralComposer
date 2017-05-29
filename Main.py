from CustomMidi.CustomTrackPool import MongoDBTrackPool
from CustomMidi.MusicianBuilder import build_musician

print("=" * 30 + "RMS_ABSOLUTE_16_8" + "=" * 30)
[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("RMS_ABSOLUTE_16_8_e"), sample_length=16,
                                                     output_length=8)

for i in range(0, 50, 1):
    musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)

print("=" * 30 + "RMS_SQARED_16_8" + "=" * 30)
[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("RMS_SQARED_16_8"), sample_length=16,
                                                     output_length=8)
for i in range(0, 50, 1):
    musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)

print("=" * 30 + "RMS_PERCENTAGE_16_8" + "=" * 30)
[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_percentage_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("RMS_PERCENTAGE_16_8"), sample_length=16,
                                                     output_length=8)
for i in range(0, 50, 1):
    musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("ADAM_ABSOLUTE_16_8"), sample_length=16,
                                                     output_length=8)
for i in range(0, 50, 1):
    musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("ADAM_SQARED_16_8"), sample_length=16,
                                                     output_length=8)
for i in range(0, 50, 1):
    musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_percentage_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("ADAM_PERCENTAGE_16_8"), sample_length=16,
                                                     output_length=8)
for i in range(0, 50, 1):
    musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("RMS_ABSOLUTE_16_4"), sample_length=16,
                                                     output_length=4)
for i in range(0, 50, 1):
    musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("RMS_SQARED_16_4"), sample_length=16,
                                                     output_length=4)
for i in range(0, 50, 1):
    musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_percentage_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("RMS_PERCENTAGE_16_4"), sample_length=16,
                                                     output_length=4)
for i in range(0, 50, 1):
    musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("ADAM_ABSOLUTE_16_4"), sample_length=16,
                                                     output_length=4)
for i in range(0, 50, 1):
    musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("ADAM_SQARED_16_4"), sample_length=16,
                                                     output_length=4)
for i in range(0, 50, 1):
    musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_percentage_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("ADAM_PERCENTAGE_16_4"), sample_length=16,
                                                     output_length=4)
for i in range(0, 50, 1):
    musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)

# musician.save(filepath="exp", overwrite=True)
