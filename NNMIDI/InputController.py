from abc import abstractmethod

import numpy as np
from mido import MidiFile


class InputTrackController:
    @abstractmethod
    def get_data_set(self):
        pass

    @abstractmethod
    def get_segment_data_set(self, start: int, length: int):
        pass

    @abstractmethod
    def build_data_set(sample_length: int, step: int, path: str, division: int) -> ([], []):
        pass


class MidiInput(InputTrackController):
    def __init__(self, division: int):
        self.division = division
        self.X = []
        self.y = []
        self.divisions = []

    def get_segment_data_set(self, start: int, length: int):
        return self.divisions[start:start + length]

    def get_data_set(self) -> ([], []):
        return [self.X, self.y]

    @staticmethod
    def build_data_set(sample_length: int, step: int, path: str, division: int) -> ([], []):
        """
        Метод-строитель класса MidiInput, определяющий датасет для обучения сети, генерирует столкьо датасетов, стколько дорожек в треке

        # ======================================================================================================================
        # Параметры такта и минимального деления
        # ======================================================================================================================
        # Семплы сообщений
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
        :rtype: [MidiInput]
        """

        midi_file = MidiFile(path)

        numerator = 4
        denominator = 4
        ticks_per_division = int(midi_file.ticks_per_beat / division * 4)

        # Время в рамках всего трека
        global_time = 0
        data_set = []

        for index, track in enumerate(midi_file.tracks):

            data_item = MidiInput(division)
            in_divisions = []
            out_divisions = []

            # Двумерный массив нажатых нот для каждой выделенной доли
            sample = []
            current = [0 for i in range(127)]

            for message in track:
                if message.is_meta and message.type == 'time_signature':
                    numerator = message.numerator
                    denominator = message.denominator

                if message.time > 0:
                    for i in range(int(message.time / ticks_per_division)):
                        sample.append(list(current))

                if message.type == 'note_on':
                    current[message.note] = 1
                if message.type == 'note_off':
                    current[message.note] = 0
                global_time += message.time

            item_index = 0

            for start in range(0, len(sample) - sample_length, step):
                buffer = []
                for i in range(start, sample_length + start):
                    buffer.append(sample[i])
                in_divisions.append(buffer)
                out_divisions.append(sample[sample_length + start])
            data_item.divisions = sample
            data_item.X = np.array(in_divisions)
            data_item.y = np.array(out_divisions)
            data_set.append(data_item)

        return data_set
