import numpy as np
from keras.layers import LSTM, Activation
from keras.models import Sequential

from NNMIDI.InputController import InputController
from NNMIDI.OutputController import OutputController


class Model(Sequential):
    """
    Наследник класса Sequential из Keras, определяет методы обучения модели и работы с данными
    """

    def __init__(self, input_: InputController, output_: OutputController):
        """
        Конструктор класса-наследика Sequential из Keras, требует обязательной инициализации контроллеров ввода/вывода
        :param input_: Контроллер ввода данных
        :type input_: InputController
        :param output_: Контроллер вывода данных
        :type output_: OutputController
        """
        super().__init__()
        self.input_controller = input_
        self.output_controller = output_

    @staticmethod
    def _threshold_sequence(prediction: list, threshold: float = 1.0) -> list:
        """
        Метод выделения звучахих нот из набора "предсказаний звучания", определяет звучащую ноту по величине вероятности звучания ноты.
        Производится бинаризация массива к виду: 1 - если больше threshold, 0 - если меньше  
        :param prediction: резульат предсказания звучащих нот
        :type prediction: list
        :param threshold: порог звучания ноты
        :type threshold: float
        :return: бинаризованный массив звучащих нот
        :rtype: list
        """
        result = []
        for i in range(len(prediction)):
            result.append(1 if prediction[i] >= threshold else 0)
        return result

    def train(self, iteration_count: int) -> None:
        """
        Специализированный метод обучения модели на данных, поступающих контроллера
        :param iteration_count: Колчиество итераций обучения
        :type iteration_count: int
        :return: None
        :rtype: None
        """
        (X, y) = self.input_controller.get_data_set()
        self.output_controller.write_log("\n  TRAINING  \n")
        self.output_controller.write_log("===========\nIterations " + str(iteration_count) + "\n" + '-' * 50 + "\n")
        self.fit(X, y, batch_size=128, epochs=iteration_count)

    def generate(self, iteration_count: int, seed: list, threshold: float = 0.7) -> tuple:
        """
        Метод генерации набора долей по сиду, составляет дорожку для трека и возвращает раздельно (сид, сгенерированная часть) 
        :param iteration_count: количество долей для генерации (желательно кратно числу долей в такте)
        :type iteration_count: int
        :param seed: входыне данные для начала генерации
        :type seed: list
        :param threshold: фильтрация звучания (смотри _threshold_sequence)
        :type threshold: float
        :return: (seed, generated)
        :rtype: tuple
        """
        iteration_seed = seed
        generated = []
        for iteration in range(iteration_count):
            raw_division = self.predict(np.array([iteration_seed]))
            self.output_controller.write_log('Raw: ' + str(raw_division))
            division = Model._threshold_sequence(raw_division[0], threshold)
            self.output_controller.write_log('Thresholded: ' + str(division))

            # TODO: Убрать отсюда!
            self.output_controller.write_data(division)

            iteration_seed.append(division)
            generated.append(division)
            iteration_seed = iteration_seed[1:]
        return seed, generated

    @staticmethod
    def build_model(input_: InputController, output_: OutputController, sample_length: int):
        """
        Метод-Строитель модели с заданными парамтерами
        :param input_: Контроллер входных данных
        :type input_: InputController
        :param output_: Контроллер выходных данных
        :type output_: OutputController
        :param sample_length: Длина единичного семпла данных
        :type sample_length: int
        :param sequence_length: Длина последовательности данных
        :type sequence_length: int
        :return: Готовая к работе необученная модель нейронной сети
        :rtype: Model
        """
        model = Model(input_, output_)
        model.add(LSTM(127, input_shape=(sample_length, 127), return_sequences=True))
        model.add(LSTM(127, input_shape=(sample_length, 127)))
        # model.add(Dense(256))
        model.add(Activation('sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model
