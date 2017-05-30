from CustomMidi.CustomTrackPool import MongoDBTrackPool
from CustomMidi.MusicianBuilder import build_musician

# print("=" * 30 + "RMS_ABSOLUTE_16_8" + "=" * 30)
# [musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
#                                                      output_pool=MongoDBTrackPool("2_RMS_ABSOLUTE_16_8_e"), sample_length=16,
#                                                      output_length=8)
# for i in range(0, 50, 1):
#     musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)
#
# print("=" * 30 + "RMS_SQARED_16_8" + "=" * 30)
# [musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
#                                                      output_pool=MongoDBTrackPool("2_RMS_SQARED_16_8"), sample_length=16,
#                                                      output_length=8)
# for i in range(0, 50, 1):
#     musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)
#
# print("=" * 30 + "RMS_PERCENTAGE_16_8" + "=" * 30)
# [musician, input_pool, output_pool] = build_musician(loss='mean_absolute_percentage_error', optimizer='RMSprop',
#                                                      output_pool=MongoDBTrackPool("2_RMS_PERCENTAGE_16_8"), sample_length=16,
#                                                      output_length=8)
# for i in range(0, 50, 1):
#     musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)
#
# [musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
#                                                      output_pool=MongoDBTrackPool("2_ADAM_ABSOLUTE_16_8"), sample_length=16,
#                                                      output_length=8)
# for i in range(0, 50, 1):
#     musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)
#
# [musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
#                                                      output_pool=MongoDBTrackPool("2_ADAM_SQARED_16_8"), sample_length=16,
#                                                      output_length=8)
# for i in range(0, 50, 1):
#     musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)
#
# [musician, input_pool, output_pool] = build_musician(loss='mean_absolute_percentage_error', optimizer='Adam',
#                                                      output_pool=MongoDBTrackPool("2_ADAM_PERCENTAGE_16_8"), sample_length=16,
#                                                      output_length=8)
# for i in range(0, 50, 1):
#     musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)
#
# [musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
#                                                      output_pool=MongoDBTrackPool("2_RMS_ABSOLUTE_16_4"), sample_length=16,
#                                                      output_length=4)
# for i in range(0, 50, 1):
#     musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)
#
# [musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
#                                                      output_pool=MongoDBTrackPool("2_RMS_SQARED_16_4"), sample_length=16,
#                                                      output_length=4)
# for i in range(0, 50, 1):
#     musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)
#
# [musician, input_pool, output_pool] = build_musician(loss='mean_absolute_percentage_error', optimizer='RMSprop',
#                                                      output_pool=MongoDBTrackPool("2_RMS_PERCENTAGE_16_4"), sample_length=16,
#                                                      output_length=4)
# for i in range(0, 50, 1):
#     musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)
#
# [musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
#                                                      output_pool=MongoDBTrackPool("2_ADAM_ABSOLUTE_16_4"), sample_length=16,
#                                                      output_length=4)
# for i in range(0, 50, 1):
#     musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)
#
# [musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
#                                                      output_pool=MongoDBTrackPool("2_ADAM_SQARED_16_4"), sample_length=16,
#                                                      output_length=4)
# for i in range(0, 50, 1):
#     musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)
#
# [musician, input_pool, output_pool] = build_musician(loss='mean_absolute_percentage_error', optimizer='Adam',
#                                                      output_pool=MongoDBTrackPool("2_ADAM_PERCENTAGE_16_4"), sample_length=16,
#                                                      output_length=4)
# for i in range(0, 50, 1):
#     musician.train(train_count=5, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("2_RMS_SQUARED_16_2"), sample_length=16,
                                                     output_length=2,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("2_RMS_SQUARED_16_4"), sample_length=16,
                                                     output_length=4,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("2_RMS_SQUARED_16_8"), sample_length=16,
                                                     output_length=8,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("2_ADAM_SQUARED_16_2"), sample_length=16,
                                                     output_length=2,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("2_ADAM_SQUARED_16_4"), sample_length=16,
                                                     output_length=4,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("2_ADAM_SQUARED_16_8"), sample_length=16,
                                                     output_length=8,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("2_RMS_ABSOLUTE_16_2"), sample_length=16,
                                                     output_length=2,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("2_RMS_ABSOLUTE_16_4"), sample_length=16,
                                                     output_length=4,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("2_RMS_ABSOLUTE_16_8"), sample_length=16,
                                                     output_length=8,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("2_ADAM_ABSOLUTE_16_2"), sample_length=16,
                                                     output_length=2,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("2_ADAM_ABSOLUTE_16_4"), sample_length=16,
                                                     output_length=4,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("2_ADAM_ABSOLUTE_16_8"), sample_length=16,
                                                     output_length=8,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

#
#
#

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("LONG_RMS_SQUARED_32_16"), sample_length=32,
                                                     output_length=16,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='RMSprop',
                                                     output_pool=MongoDBTrackPool("LONG_RMS_ABSOLUTE_32_16"), sample_length=32,
                                                     output_length=16,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_squared_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("LONG_ADAM_SQUARED_32_16"), sample_length=32,
                                                     output_length=16,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

[musician, input_pool, output_pool] = build_musician(loss='mean_absolute_error', optimizer='Adam',
                                                     output_pool=MongoDBTrackPool("LONG_ADAM_ABSOLUTE_32_16"), sample_length=32,
                                                     output_length=16,
                                                     threshold_delta=0.00009
                                                     )
for i in range(0, 20, 1):
    musician.train(train_count=1, input_pool=input_pool, output_pool=output_pool)

musician.save(filepath="exp", overwrite=True)
