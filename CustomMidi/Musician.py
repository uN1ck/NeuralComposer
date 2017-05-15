import random

import numpy as np
from keras.models import Sequential

from CustomMidi.CustomTrack import CustomTrack


class Musician(Sequential):
    """
    Наследник класса Sequential из Keras, определяет методы обучения модели и работы с данными
    """

    def __init__(self, track_in: CustomTrack, step_length: int):
        """
        Конструктор класса-наследика Sequential из Keras, требует обязательной инициализации контроллеров ввода/вывода
        """
        super().__init__()
        self.track_in = track_in
        self.step_length = step_length

    @staticmethod
    def _threshold_sequence_max_delta(prediction: list, delta: float = 0.09, ) -> list:
        """
        Метод выделения звучахих нот из набора "предсказаний звучания", определяет звучащую ноту по величине вероятности звучания ноты.
        Производится бинаризация массива к виду: 1 - если отличается от max на delta, 0 - если иначе  
        :param prediction: резульат предсказания звучащих нот
        :param delta: порог звучания ноты
        :return: бинаризованный массив звучащих нот
        """
        result = []
        max_val = max(prediction)
        for i in range(len(prediction)):
            result.append(1 if max_val - prediction[i] <= delta else 0)
        return result

    def train(self, iteration_count: int) -> None:
        """
        Специализированный метод обучения модели на данных, поступающих контроллера
        :param iteration_count: Колчиество итераций обучения
        :return: None
        """
        (X, y) = self.track_in.get_data_set(1, self.step_length)
        # self.output_controller.write_log("\n  TRAINING  \n")
        # self.output_controller.write_log("===========\nIterations " + str(iteration_count) + "\n" + '-' * 50 + "\n")
        self.fit(X, y, batch_size=128, epochs=iteration_count)

    def generate(self, seed: list, iteration_count: int) -> tuple:
        """
        Метод генерации набора долей по сиду, составляет дорожку для трека и возвращает раздельно (сид, сгенерированная часть) 
        :param seed: входыне данные для начала генерации
        :param iteration_count: количество долей для генерации (желательно кратно числу долей в такте)
        :param threshold: фильтрация звучания (смотри _threshold_sequence)
        :return: (seed, generated)
        """
        iteration_seed = seed
        generated = []
        raw = []
        for iteration in range(iteration_count):
            raw_division = self.predict(np.array([iteration_seed]))
            raw.append(raw_division)
            division = Musician._threshold_sequence_max_delta(raw_division[0])  # , threshold)

            iteration_seed.append(division)
            generated.append(division)
            iteration_seed = iteration_seed[1:]

        f = open("log" + str(random.randint(0, 100000000)) + ".txt", "w")
        f.write("=" * 20 + "INSTRUMENT!" + "=" * 20 + "\n")
        for line in raw:
            for item in line:
                f.write(str(item) + " ")
            f.write("\n")
        f.close()

        return seed, generated
