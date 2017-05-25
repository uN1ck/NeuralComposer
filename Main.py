from CustomMidi.MusicianBuilder import build_musician

(musician, input_pool, output_pool) = build_musician()
for i in range(20):
    musician.train(train_count=2, input=input_pool, output=output_pool)
