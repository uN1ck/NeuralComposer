import numpy as np
from mido import MidiFile


class CustomTrack:
    """
    Класс отвечающий за конкретную дорожку в треке, хранит ее в разобранном виде c заданнйо частотой разбиения
    """

    def __init__(self, division: int):
        """
        Конструктор трека
        :param division: Частота разбиения трека
        """
        self.divisions = []
        self.division = division

    def get_data_set(self, step: int, sample_length: int, result_length: int = 1) -> [np.array, np.array]:
        """
        Метод доступа к подготовленному датасету, при каждом вызове формирует его заново. 
        :param step: Длина смещения в долях минимального разбиения
        :param sample_length: Длина единичного семпла для входных данных обучения
        :param result_length: Длина единичного семпла для выходных данных обучения, как правило равна 1
        :return: Пара массивов [X, y], где X входные, а y - выходные
        """
        in_divisions = []
        out_divisions = []

        for start in range(0, len(self.divisions) - sample_length, step):
            buffer = []
            for i in range(start, sample_length + start):
                buffer.append(self.divisions[i])
            in_divisions.append(buffer)
            out_divisions.append(self.divisions[sample_length + start])
        return [np.array(in_divisions), np.array(out_divisions)]

    def get_segment_data_set(self, start: int, length: int) -> [np.array]:
        """
        Метод доступа к сегменту из датасета
        :param start: Индекс первой доли сегмента в датасете
        :param length: Длина сегмента
        :return: Массив долей заданной длины
        """
        # TODO: Ошибки на размер датасета?
        return np.array(self.divisions[start:start + length])


class CustomMidiFile:
    """
    Миди файл. Класс-контейнер треков, определяет темп, размер и профчие характеристики трека
    """

    def __init__(self):
        self.tracks = []
        self.numerator = 4
        self.denominator = 4

    @staticmethod
    def build_data_set(sample_length: int, step: int, path: str, division: int) -> ([], []):
        """
        Метод-строитель класса MidiInput, определяющий датасет для обучения сети, генерирует столкьо датасетов, стколько дорожек в треке

        # ======================================================================================================================
        # Семплы сообщений
        # ======================================================================================================================
        # <meta message time_signature numerator=4 denominator=4 clocks_per_click=24 notated_32nd_notes_per_beat=8 time=0>
        # note_on channel=0 note=45 velocity=80 time=0
        # note_off channel=0 note=45 velocity=64 time=480
        # ======================================================================================================================

        :param division: Минимальная доля разбиения
        :type division: int
        :param step: Шаг смещения при построении, отвечает за смещение окна входных данных (семпла) от предыдущего положения 
            (для создания высокой связаности лучше использовать 1)
        :type step: int
        :param sample_length: Число долей, входящее в окно входных данных (семпл), по логике должно быть кратно количеству долей в такте
        :type sample_length: int
        :param path: Путь к файлу mid
        :type path: str
        :return: Массив MidiInput, где каждый отдельный экземпляр определяет отдельную дорожку трека
        """

        new_midi_file = CustomMidiFile()
        midi = MidiFile(path)
        ticks_per_division = int(midi.ticks_per_beat / division * 4)
        global_time = 0

        for index, track in enumerate(midi.tracks):
            data_item = CustomTrack(division)
            # Двумерный массив нажатых нот для каждой выделенной доли
            sample = []
            current = [0 for i in range(127)]

            for message in track:
                if message.is_meta and message.type == 'time_signature':
                    new_midi_file.numerator = message.numerator
                    new_midi_file.denominator = message.denominator

                if message.time > 0:
                    for i in range(int(message.time / ticks_per_division)):
                        sample.append(list(current))

                if message.type == 'note_on':
                    current[message.note] = 1
                if message.type == 'note_off':
                    current[message.note] = 0
                global_time += message.time
            data_item.divisions = sample
            new_midi_file.tracks.append(data_item)
        return new_midi_file.tracks
