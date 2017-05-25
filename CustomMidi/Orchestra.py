from keras.layers import LSTM, Conv1D, Dropout

from CustomMidi.CustomTrack import CustomTrack
from CustomMidi.CustomTrackPool import CustomTrackPoolInterface, MongoDBTrackPool
from CustomMidi.Musician import Musician


class Orchestra:
    def __init__(self, input: CustomTrackPoolInterface, output: CustomTrackPoolInterface):
        self.musicians = []
        self.input = input
        self.output = output

    def train(self, train_counts: int):
        for track in self.input:
            for musician in self.musicians:
                musician.train(train_counts, track)
            # TODO: Нормальные логи генерации а не вот это вот все!
            # =======================================================================================================
            self.generate(track.get_segment_data_set(0, self.musicians[0].x_size), 128, track.name)
            # =======================================================================================================

    def generate(self, seed: list, length: int, name: str):
        for musician in self.musicians:
            (seed, generated, raw) = musician._generate(seed, length)
            self.output.put_track(CustomTrack(8, 4, 4, generated), name, raw)


def build_orchestra(division_in: int, division_out: int, sample_length: int,
                    output_length: int, thresholder, loss, optimizer):
    # TODO: сделать для несокльких инпутов несоклько музыкантов!

    input_file = MongoDBTrackPool()
    output_file = MongoDBTrackPool()
    result = Orchestra(input_file, output_file)

    musician = build_musician(sample_length, output_length, thresholder, loss, optimizer)
    result.musicians.append(musician)

    return result


def build_musician(sample_length: int, output_length: int, thresholder, loss='mean_squared_error',
                   optimizer='RMSprop') -> Musician:
    # TODO: сделать для несокльких инпутов!
    # TODO: сделать расчет на правильное создание сверток!
    model = Musician(sample_length, output_length, thresholder)

    # model.add(LSTM(127, input_shape=(sample_length, 127)))
    model.add(Conv1D(input_shape=(sample_length, 127), filters=127, kernel_size=1, strides=int(sample_length / output_length)))
    model.add(Dropout(input_shape=(sample_length, 127), rate=0.2))
    model.add(LSTM(127, return_sequences=True))
    # model.add(LSTM(127))

    model.compile(loss=loss, optimizer=optimizer)

    return model
