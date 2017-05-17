from keras.layers import LSTM

from CustomMidi.CustomTrack import CustomTrack
from CustomMidi.CustomTrackPool import CustomTrackPool
from CustomMidi.Musician import Musician


class Orchestra:
    def __init__(self, input: CustomTrackPool, output: CustomTrackPool):
        self.musicians = []
        self.input = input
        self.output = output

    def train(self, train_counts: int):
        for track in self.input.data_pool:
            for musician in self.musicians:
                musician.train(train_counts, track)
            # TODO: Нормальные логи генерации а не вот это вот все!
            # =======================================================================================================
            self.generate(track.get_segment_data_set(0, self.musicians[0].x_size), 32)
            # =======================================================================================================
        self.output.build_midi_files("res")

    def generate(self, seed: list, length: int):
        for musician in self.musicians:
            (seed, generated) = musician.generate(seed, length)
            self.output.data_pool.append(CustomTrack(8, 4, 4, generated))


def build_orchestra(input_path: str, output_path: str, division_in: int, division_out: int, sample_length: int):
    # TODO: сделать для несокльких инпутов!
    input_file = CustomTrackPool(input_path)
    output_file = CustomTrackPool(path_to_data_pool=None)
    result = Orchestra(input_file, output_file)

    musician = build_musician(sample_length)
    result.musicians.append(musician)

    return result


def build_musician(sample_length: int) -> Musician:
    # TODO: сделать для несокльких инпутов!
    output_length = 1
    model = Musician(sample_length, output_length)
    model.add(LSTM(127, input_shape=(sample_length, 127), return_sequences=True))
    model.add(LSTM(127, input_shape=(sample_length, 127)))

    # model.add(Conv1D(filters=127, kernel_size=1, strides=int(sample_length / output_length)))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
