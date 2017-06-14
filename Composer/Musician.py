import numpy as np
from keras.models import Sequential, load_model

from Composer.CustomTrack import CustomTrack1D
from Composer.CustomTrackPool import CustomTrackPoolInterface


class Musician:
    """
    Обертка класса Sequential из Keras, определяет методы обучения модели и работы с данными
    """

    def __init__(self, x_size: int, y_size: int, batch_size: int = 32, model: Sequential = Sequential()):
        """
        Конструктор класса-наследика Sequential из Keras, требует обязательной инициализации контроллеров ввода/вывода
        """
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.batch_size = batch_size
        self.model = model

    @staticmethod
    def threshold_sequence_max_delta(data_set: list) -> list:
        """Метод выделения звучахих нот из набора "предсказаний звучания",
        определяет звучащую ноту по величине вероятности звучания ноты.
        Производится бинаризация массива к виду: 1 - если отличается от max на delta, 0 - если иначе
        :param data_set: резульат предсказания звучащих нот
        :param delta: ???
        :return: бинаризованный массив звучащих нот
        """

        result = []
        for item in data_set:

            maxed = list(set(sorted(item, reverse=True)))
            if len(maxed) > 1:
                delta = abs(maxed[0] - maxed[1])
            else:
                delta = maxed[0]

            max_val = maxed[0]
            buffer = []
            max_val = max(item)
            for i in range(len(item)):
                buffer.append(1 if abs(max_val - item[i]) < delta else 0)
            result.append(buffer)
        return result

    def train(self, train_count: int, input_pool: CustomTrackPoolInterface, output_pool: CustomTrackPoolInterface,
              is_logged: bool = True):
        """
        Специализированный метод обучения модели на данных, поступающих контроллера.
        Генерирует логи после каждой итерации обучения состоящей из числа epochs обучений
        :param is_logged: Параметр, определяющйи требуется ли наличие логов генерации в БД
        :param input_pool: Интерфейс входных данных для модели
        :param output_pool: Интерфейс выходных данных для модели, используется для логирования
        :param train_count: Количество итераций обучения
        :return: None
        """
        for track in input_pool:
            (X, y) = track.get_data_set(1, self.x_size, self.y_size)
            try:
                self.model.fit(x=X, y=y, batch_size=self.batch_size, epochs=train_count, verbose=1)
            except Exception as ex:
                print(ex)

            if is_logged:
                # =======================================================================================================
                self.generate(seed=track.get_segment_data_set(0, self.x_size), iteration_count=256, name=track.name,
                              output=output_pool)
                # =======================================================================================================

    def generate(self, seed: list, iteration_count: int, name: str, output: CustomTrackPoolInterface,
                 track: CustomTrack1D = CustomTrack1D(8, 4, 4, [], "")) -> tuple:
        """Метод генерации набора долей по сиду, составляет дорожку для трека и
        возвращает раздельно (сид, сгенерированная часть)
        :param track: Экземпляр Track, с заданными параметрами размера и разбиения, 
            используется в качестве контейнера сгененрированных данных для дальнейшей передачи в TrackPoolDoge
        :param output: Интерфейс выходных данных для модели
        :param name: имя трека при генерации. Присваивается треку для логирования, например, по принадлежности к сиду его же именем.
        :param seed: входыне данные для начала генерации
        :param iteration_count: количество долей для генерации (желательно кратно числу долей в такте),
            должно быть больше чем y_size!
        :return: (seed, generated, raw)
        """
        iteration_seed = seed
        generated = []
        raw = []
        for iteration in range(int(iteration_count / self.y_size)):
            raw_division = self.model.predict(np.array([iteration_seed]), self.batch_size)[0].tolist()
            raw += raw_division
            division = []

            division += self.threshold_sequence_max_delta(raw_division)

            iteration_seed += division
            generated += division
            iteration_seed = iteration_seed[self.y_size:]

        track.divisions = generated
        track.name = name

        if output is not None:
            print("Saving in output")
            output.put_track(track, raw)
        else:
            print("No output!")
        return seed, generated, raw

    def save(self, filepath, overwrite=True, include_optimizer=True):
        self.model.save(filepath, overwrite, include_optimizer)

    @staticmethod
    def load(filepath):
        model = load_model(filepath)
        x = model.input_shape[1]
        y = model.output_shape[1]
        musician = Musician(x, y, model=model)
        return musician
