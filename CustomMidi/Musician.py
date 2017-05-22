import numpy as np
from keras.models import Sequential

from CustomMidi.CustomTrack import CustomTrack


class Musician(Sequential):
    """
    Наследник класса Sequential из Keras, определяет методы обучения модели и работы с данными
    """

    def __init__(self, x_size: int, y_size: int):
        """
        Конструктор класса-наследика Sequential из Keras, требует обязательной инициализации контроллеров ввода/вывода
        """
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size

    @staticmethod
    def _threshold_sequence_max_delta(value: list, delta: float = 0.09, ) -> list:
        """
        Метод выделения звучахих нот из набора "предсказаний звучания", определяет звучащую ноту по величине вероятности звучания ноты.
        Производится бинаризация массива к виду: 1 - если отличается от max на delta, 0 - если иначе  
        :param prediction: резульат предсказания звучащих нот
        :param delta: порог звучания ноты
        :return: бинаризованный массив звучащих нот
        """
        result = []
        max_val = max(value)
        for i in range(len(value)):
            result.append(1 if max_val - value[i] <= delta else 0)
        return result

    def train(self, iteration_count: int, data_set: CustomTrack) -> None:
        """
        Специализированный метод обучения модели на данных, поступающих контроллера
        :param data_set: 
        :param iteration_count: Колчиество итераций обучения
        :return: None
        """
        (X, y) = data_set.get_data_set(1, self.x_size, self.y_size)
        self.fit(X, y, batch_size=128, epochs=iteration_count)

    def generate(self, seed: list, iteration_count: int) -> tuple:
        """
        Метод генерации набора долей по сиду, составляет дорожку для трека и возвращает раздельно (сид, сгенерированная часть) 
        :param threshold_function: Функция определяющая нажаты енот ыв массиве предсказаний для доли. Обязательно должна иметь 
        параметром value в который будет передаваться необработанная доля
        :param seed: входыне данные для начала генерации
        :param iteration_count: количество долей для генерации (желательно кратно числу долей в такте)
        :return: (seed, generated, raw)
        """
        iteration_seed = seed
        generated = []
        raw = []
        for iteration in range(iteration_count):
            raw_division = self.predict(np.array([iteration_seed]))[0].tolist()
            raw += raw_division
            division = []
            for division_item in raw_division:
                division.append(Musician._threshold_sequence_max_delta(division_item))  # , threshold)

            iteration_seed += division
            generated += division
            iteration_seed = iteration_seed[self.y_size:]

        return seed, generated, raw
