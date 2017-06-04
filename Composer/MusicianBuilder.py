from keras.layers import LSTM, Dropout
from keras.models import Sequential

from Composer.CustomTrackPool import CustomTrackPoolInterface, MongoDBTrackPool
from Composer.Musician import Musician


def build_musician(input_pool: CustomTrackPoolInterface = None,
                   output_pool: CustomTrackPoolInterface = None,
                   sample_length: int = 32,
                   tracks_count: int = 1,
                   threshold_delta=256,
                   loss='mean_absolute_error',
                   optimizer='RMSprop') -> [Musician, CustomTrackPoolInterface, CustomTrackPoolInterface]:
    """
    Фабричный метод создания модели ИНС. по заданным параметрам строит сеть 
    :param tracks_count: Количество треков при обучении и генерации сети, которые следует генерировать.
        В случае если равно 1 (по умолчанию) то все треки сливаются в 1. 
        Во всех других следует подгатавидивать midi-данные на заданное число треков.
    :param input_pool: ?
    :param output_pool:  ?
    :param sample_length: Количество долей, поступающее на вход при каждой итерации обучения
    :param threshold_delta: Функция приведения выходных данных нейросети к требуемому виду.
        Обязательно наличие параметра value, с который будут передаваться выходные данные. 
    :param loss: Функция потерь см. Keras
    :param optimizer: Оптимизатор компиляции см. Keras
    :return: Musician - собранная, но необученная модель. 
    """
    if input_pool is None:
        input_pool = MongoDBTrackPool("TrainSet")
    if output_pool is None:
        output_pool = MongoDBTrackPool("ResultSet")

    # TODO: Дописать на множесто треков merge архитектуру???
    model = Sequential()

    model.add(LSTM(127, input_shape=(sample_length, 127), return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(127, return_sequences=True))

    model.compile(loss=loss, optimizer=optimizer)
    musician = Musician(model.input_shape[1], model.output_shape[1], threshold_delta, model=model)
    return [musician, input_pool, output_pool]
