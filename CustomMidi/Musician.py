import numpy as np
from keras.models import Sequential

from CustomMidi.CustomTrack import CustomTrack
from CustomMidi.CustomTrackPool import CustomTrackPoolInterface


# ================================================================================================================================
# Функции для подготовки данных (трешхолдеры), поступающх из нейросети, используется при КАЖДОЙ генерации
# ================================================================================================================================
def threshold_sequence_max_delta(data_set: list, delta: float = 0.0009) -> list:
    """
    Метод выделения звучахих нот из набора "предсказаний звучания", определяет звучащую ноту по величине вероятности звучания ноты.
    Производится бинаризация массива к виду: 1 - если отличается от max на delta, 0 - если иначе  
    :param data_set: резульат предсказания звучащих нот
    :param delta: порог звучания ноты
    :return: бинаризованный массив звучащих нот
    """
    result = []
    for item in data_set:
        buffer = []
        max_val = max(item)
        for i in range(len(item)):
            buffer.append(1 if abs(max_val - item[i]) < delta else 0)
        result.append(buffer)
    return result


def threshold_sequence_max(data_set: list) -> list:
    """
    Метод выделения звучахих нот из набора "предсказаний звучания", определяет звучащую ноту по величине вероятности звучания ноты.
    Для каждой доли разделения выбирает только максимальныее значения
    :param data_set: резульат предсказания звучащих нот
    :param delta: порог звучания ноты
    :return: бинаризованный массив звучащих нот
    """
    result = []
    for item in data_set:
        buffer = []
        max_val = max(item)
        for i in range(len(item)):
            buffer.append(1 if max_val == item[i] else 0)
        result.append(buffer)
    return result


# ================================================================================================================================
# Функции для подготовки данных (трешхолдеры), поступающх из нейросети
# ================================================================================================================================
class Musician(Sequential):
    """
    Наследник класса Sequential из Keras, определяет методы обучения модели и работы с данными
    """

    def __init__(self, x_size: int, y_size: int, threshold_function=threshold_sequence_max_delta, batch_size: int = 32):
        """
        Конструктор класса-наследика Sequential из Keras, требует обязательной инициализации контроллеров ввода/вывода
        """
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.batch_size = batch_size
        self.threshold_function = threshold_function

    def train(self, train_count: int, input: CustomTrackPoolInterface, output: CustomTrackPoolInterface):
        """
        Специализированный метод обучения модели на данных, поступающих контроллера.
        Генерирует логи после каждой итерации обучения состоящей из числа epochs обучений
        :param input: Интерфейс входных данных для модели
        :param output: Интерфейс выходных данных для модели, используется для логирования
        :param train_count: Количество итераций обучения
        :return: None
        """
        for track in input:
            (X, y) = track.get_data_set(1, self.x_size, self.y_size)
            self.fit(x=X, y=y, batch_size=self.batch_size, epochs=train_count, verbose=1)

            # TODO: Нормальные логи генерации а не вот это вот все!
            # =======================================================================================================
            self.generate(seed=track.get_segment_data_set(0, self.x_size), iteration_count=16, name=track.name, output=output)
            # =======================================================================================================

    def generate(self, seed: list, iteration_count: int, name: str, output: CustomTrackPoolInterface,
                 track: CustomTrack = CustomTrack(8, 4, 4, [], "")) -> tuple:
        """
        Метод генерации набора долей по сиду, составляет дорожку для трека и возвращает раздельно (сид, сгенерированная часть) 
        :param track: Экземпляр Track, с заданными параметрами размера и разбиения, 
            используется в качестве контейнера сгененрированных данных для дальнейшей передачи в TrackPool
        :param output: Интерфейс выходных данных для модели
        :param name: имя трека при генерации. Присваивается треку для логирования, например, по принадлежности к сиду его же именем.
        :param seed: входыне данные для начала генерации
        :param iteration_count: количество долей для генерации (желательно кратно числу долей в такте)
        :return: (seed, generated, raw)
        """
        iteration_seed = seed
        generated = []
        raw = []
        for iteration in range(iteration_count):
            raw_division = self.predict(np.array([iteration_seed]), self.batch_size)[0].tolist()
            raw += raw_division
            division = []

            division += self.threshold_function(raw_division)

            iteration_seed += division
            generated += division
            iteration_seed = iteration_seed[self.y_size:]

        track.divisions = generated
        track.name = name

        output.put_track(track, raw)
        return seed, generated, raw
