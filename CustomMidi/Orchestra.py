from keras.layers import LSTM, Conv1D

from CustomMidi.CustomTrack import CustomTrack
from CustomMidi.CustomTrackPool import CustomTrackPool, CustomTrackPoolInterface
from CustomMidi.Musician import Musician


class Orchestra:
    def __init__(self, input: CustomTrackPoolInterface, output: CustomTrackPoolInterface):
        self.musicians = []
        self.input = input
        self.output = output

    def train(self, train_counts: int):
        for track in self.input.get_data_pool():
            for musician in self.musicians:
                musician.train(train_counts, track)
            # TODO: Нормальные логи генерации а не вот это вот все!
            # =======================================================================================================
            self.generate(track.get_segment_data_set(0, self.musicians[0].x_size), 128)
            # =======================================================================================================
            # self.output.build_midi_files("res")

    def generate(self, seed: list, length: int):
        for musician in self.musicians:
            (seed, generated) = musician.generate(seed, length)
            self.output.put_track(CustomTrack(8, 4, 4, generated))


def build_orchestra(input_path: str, output_path: str, division_in: int, division_out: int, sample_length: int,
                    output_length: int):
    # TODO: сделать для несокльких инпутов!
    input_file = CustomTrackPool(input_path, division_in)
    output_file = CustomTrackPool(path_to_data_pool=None, division=division_out)
    result = Orchestra(input_file, output_file)

    musician = build_musician(sample_length, output_length)
    result.musicians.append(musician)

    return result


def build_musician(sample_length: int, output_length: int) -> Musician:
    # TODO: сделать для несокльких инпутов!
    model = Musician(sample_length, output_length)

    # model.add(LSTM(127, input_shape=(sample_length, 127)))
    model.add(Conv1D(input_shape=(sample_length, 127), filters=127, kernel_size=1, strides=int(sample_length / output_length)))
    model.add(LSTM(127, return_sequences=True))
    # model.add(LSTM(127))

    model.compile(loss='kullback_leibler_divergence', optimizer='Nadam')
    return model
