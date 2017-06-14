from Composer.CustomTrackPool import MongoDBTrackPool
from Composer.MusicianBuilder import build_musician

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("3_CNNEDE_RMS_SQUARED_8"), sample_length=32,
                                                     filters=8
                                                     )
musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_CNNEDE_RMS_SQUARED_8", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("3_CNNEDE_RMS_SQUARED_16"), sample_length=32,
                                                     filters=16
                                                     )
musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_CNNEDE_RMS_SQUARED_16", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("3_CNNEDE_RMS_SQUARED_32"), sample_length=32,
                                                     filters=32
                                                     )
musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_CNNEDE_RMS_SQUARED_32", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("3_CNNEDE_RMS_SQUARED_64"), sample_length=32,
                                                     filters=64
                                                     )
musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_CNNEDE_RMS_SQUARED_64", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("3_CNNEDE_ADAM_SQUARED_8"), sample_length=32,
                                                     filters=8
                                                     )
musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_CNNEDE_ADAM_SQUARED_8", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("3_CNNEDE_ADAM_SQUARED_16"), sample_length=32,
                                                     filters=16
                                                     )
musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_CNNEDE_ADAM_SQUARED_16", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("3_CNNEDE_ADAM_SQUARED_32"), sample_length=32,
                                                     filters=32
                                                     )
musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_CNNEDE_ADAM_SQUARED_32", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("3_CNNEDE_ADAM_SQUARED_64"), sample_length=32,
                                                     filters=64
                                                     )
musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_CNNEDE_ADAM_SQUARED_64", overwrite=True)

'''
[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("3_RMS_SQUARED_16_hs"), sample_length=16,
                                                     activation='hard_sigmoid'
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_RMS_SQUARED_16_hs", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("3_RMS_SQUARED_16_t"), sample_length=16,
                                                     activation='tanh'
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_RMS_SQUARED_16_t", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("3_ADAM_SQUARED_16_s"), sample_length=16,
                                                     activation='sigmoid'
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_ADAM_SQUARED_16_s", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("3_ADAM_SQUARED_16_hs"), sample_length=16,
                                                     activation='hard_sigmoid'
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_ADAM_SQUARED_16_hs", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("3_ADAM_SQUARED_16_t"), sample_length=16,
                                                     activation='tanh'
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_ADAM_SQUARED_16_t", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("3_RMS_ABSOLUTE_16_s"), sample_length=16,
                                                     activation='sigmoid'
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_RMS_ABSOLUTE_16_s", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("3_RMS_ABSOLUTE_16_hs"), sample_length=16,
                                                     activation='hard_sigmoid'
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_RMS_ABSOLUTE_16_hs", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("3_RMS_ABSOLUTE_16_t"), sample_length=16,
                                                     activation='tanh'
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_RMS_ABSOLUTE_16_t", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("3_ADAM_ABSOLUTE_16_s"), sample_length=16,
                                                     activation='sigmoid'
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_ADAM_ABSOLUTE_16_s", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("3_ADAM_ABSOLUTE_16_hs"), sample_length=16,
                                                     activation='hard_sigmoid'
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_ADAM_ABSOLUTE_16_hs", overwrite=True)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("3_ADAM_ABSOLUTE_16_t"), sample_length=16,
                                                     activation='tanh'
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=10, input_pool=input_pool, output_pool=output_pool)
musician.save(filepath="3_ADAM_ABSOLUTE_16_t", overwrite=True)
'''
