import numpy as np
from keras.layers import LSTM, Activation
from keras.models import Sequential

from CustomMidi.CustomMidiFile import CustomMidiFile
from CustomMidi.CustomTrack import CustomTrack


class Model(Sequential):
    """
    Наследник класса Sequential из Keras, определяет методы обучения модели и работы с данными
    """

    def __init__(self, midi_in: CustomTrack, step_length: int):
        """
        Конструктор класса-наследика Sequential из Keras, требует обязательной инициализации контроллеров ввода/вывода
        """
        super().__init__()
        self.midi_in = midi_in
        self.step_length = step_length

    @staticmethod
    def _threshold_sequence(prediction: list, threshold: float = 1.0) -> list:
        """
        Метод выделения звучахих нот из набора "предсказаний звучания", определяет звучащую ноту по величине вероятности звучания ноты.
        Производится бинаризация массива к виду: 1 - если больше threshold, 0 - если меньше  
        :param prediction: резульат предсказания звучащих нот
        :param threshold: порог звучания ноты
        :return: бинаризованный массив звучащих нот
        """
        result = []
        for i in range(len(prediction)):
            result.append(1 if prediction[i] >= threshold else 0)
        return result

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
        (X, y) = self.midi_in.get_data_set(1, self.step_length)
        # self.output_controller.write_log("\n  TRAINING  \n")
        # self.output_controller.write_log("===========\nIterations " + str(iteration_count) + "\n" + '-' * 50 + "\n")
        self.fit(X, y, batch_size=128, epochs=iteration_count)

    def generate(self, iteration_count: int, seed: list, threshold: float = 0.7) -> tuple:
        """
        Метод генерации набора долей по сиду, составляет дорожку для трека и возвращает раздельно (сид, сгенерированная часть) 
        :param iteration_count: количество долей для генерации (желательно кратно числу долей в такте)
        :param seed: входыне данные для начала генерации
        :param threshold: фильтрация звучания (смотри _threshold_sequence)
        :return: (seed, generated)
        """
        # TODO: Добавить логи в текст или убрать?
        iteration_seed = seed
        generated = []
        for iteration in range(iteration_count):
            raw_division = self.predict(np.array([iteration_seed]))
            division = Model._threshold_sequence_max_delta(raw_division[0])  # , threshold)

            iteration_seed.append(division)
            generated.append(division)
            iteration_seed = iteration_seed[1:]
        return seed, generated


def build_model(midi_file_in: CustomMidiFile, sample_length: int) -> Model:
    """
    Метод-Строитель модели с заданными парамтерами
    
    Находится в тестовом соостоянии, в миди-файлах используетс ятолкьо 0й трек!
    
    :param midi_file_in: Файл миди, используемый для даннных обучения
    :param midi_file_out: файл миди, используемый для данных генерации
    :param sample_length: Длина единичного семпла данных
    :return: Готовая к работе необученная модель нейронной сети
    """
    # TODO: Собиралка набора моделей из миди файла, пока что костыль на нулевой трек
    model = Model(midi_file_in.tracks[0], sample_length)

    model.add(LSTM(127, input_shape=(sample_length, 127), return_sequences=True))
    model.add(LSTM(127, input_shape=(sample_length, 127)))
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='Nadam')
    return model
