from Composer.CustomTrackPool import MongoDBTrackPool
from Composer.MusicianBuilder import build_musician

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("2_RMS_SQUARED_16_2"), sample_length=16,
                                                     output_length=2,
                                                     threshold_delta=0.0009
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="2_RMS_SQUARED_16_2", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("2_RMS_SQUARED_16_4"), sample_length=16,
                                                     output_length=4,
                                                     threshold_delta=0.0009
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="2_RMS_SQUARED_16_4", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("2_RMS_SQUARED_16_8"), sample_length=16,
                                                     output_length=8,
                                                     threshold_delta=0.0009
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="2_RMS_SQUARED_16_8", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("2_ADAM_SQUARED_16_2"), sample_length=16,
                                                     output_length=2,
                                                     threshold_delta=0.0009
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="2_ADAM_SQUARED_16_2", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("2_ADAM_SQUARED_16_4"), sample_length=16,
                                                     output_length=4,
                                                     threshold_delta=0.0009
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="2_ADAM_SQUARED_16_4", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("2_ADAM_SQUARED_16_8"), sample_length=16,
                                                     output_length=8,
                                                     threshold_delta=0.0009
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="2_ADAM_SQUARED_16_8", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("2_RMS_ABSOLUTE_16_2"), sample_length=16,
                                                     output_length=2,
                                                     threshold_delta=0.0009
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="2_RMS_ABSOLUTE_16_2", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("2_RMS_ABSOLUTE_16_4"), sample_length=16,
                                                     output_length=4,
                                                     threshold_delta=0.0009
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="2_RMS_ABSOLUTE_16_4", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("2_RMS_ABSOLUTE_16_8"), sample_length=16,
                                                     output_length=8,
                                                     threshold_delta=0.0009
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="2_RMS_ABSOLUTE_16_8", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("2_ADAM_ABSOLUTE_16_2"), sample_length=16,
                                                     output_length=2,
                                                     threshold_delta=0.0009
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="2_ADAM_ABSOLUTE_16_2", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("2_ADAM_ABSOLUTE_16_4"), sample_length=16,
                                                     output_length=4,
                                                     threshold_delta=0.0009
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="2_ADAM_ABSOLUTE_16_4", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("2_ADAM_ABSOLUTE_16_8"), sample_length=16,
                                                     output_length=8,
                                                     threshold_delta=0.0009
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="2_ADAM_ABSOLUTE_16_8", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("LONG_RMS_SQUARED_32_16"), sample_length=32,
                                                     output_length=16,
                                                     threshold_delta=0.00005
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="LONG_RMS_SQUARED_32_16", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("LONG_RMS_ABSOLUTE_32_16"), sample_length=32,
                                                     output_length=16,
                                                     threshold_delta=0.00005
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="LONG_RMS_ABSOLUTE_32_16", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("LONG_ADAM_SQUARED_32_16"), sample_length=32,
                                                     output_length=16,
                                                     threshold_delta=0.00005
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="LONG_ADAM_SQUARED_32_16", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("LONG_ADAM_ABSOLUTE_32_16"), sample_length=32,
                                                     output_length=16,
                                                     threshold_delta=0.00005
                                                     )
for i in range(0, 40, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
    musician.save(filepath="LONG_ADAM_ABSOLUTE_32_16", overwrite=True)
