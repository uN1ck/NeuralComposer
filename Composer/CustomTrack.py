from abc import abstractmethod

import mido
import numpy as np
from mido import MidiFile, MidiTrack, Message
from mido.midifiles.midifiles import DEFAULT_TICKS_PER_BEAT


# ================================================================================================================================
# Интерфейс для пула треков
# ================================================================================================================================
class CustomTrackInterface:
    def __init__(self, division: int, numerator: int, denominator: int, divisions: list = list, name: str = "default"):
        self.divisions = divisions
        self.division = division
        self.numerator = numerator
        self.denominator = denominator
        self.name = name

    @abstractmethod
    def get_data_set(self, step: int, sample_length: int, result_length: int = 1) -> [list, list]:
        pass;

    @abstractmethod
    def get_segment_data_set(self, start: int, length: int) -> list:
        pass;

    def parse_midi_file(self, midi_file: MidiFile) -> None:
        """
        Метод инициализации обертки из файла. Собирает все данные с файла и сливает их в один трек.
        Ударные не исключаются! После слияния разбивает трек на "минимальные заданные доли" и
        собирает набор звучащих нот
        Args:
            midi_file (MidiFile): Произвольный MIDI-файл
        Returns:
            None
        """
        merged_track = mido.merge_tracks(midi_file.tracks)
        ticks_per_division = int(midi_file.ticks_per_beat / self.division * self.numerator)
        global_time = 0

        sample = []
        current = [0 for i in range(127)]

        for message in merged_track:
            if message.is_meta and message.type == 'time_signature':
                self.numerator = message.numerator
                self.denominator = message.denominator

            if message.time > 0:
                for i in range(int(message.time / ticks_per_division)):
                    sample.append(list(current))

            if message.type == 'note_on':
                current[message.note] = 1
            if message.type == 'note_off':
                current[message.note] = 0
            global_time += message.time
        self.divisions = sample

    def build_midi_file(self, name: str, numerator: int, denominator: int,
                        ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT) -> None:
        """Метод  построения midi-файла из набора треков экземпляра класса.
        :param ticks_per_beat:
        :param name: Названеи midi-файла, в который будет произведеня сборка
        :param numerator: Количество долей
        :param denominator: Длительность доли
        :return: None
        """
        midi = MidiFile(ticks_per_beat=ticks_per_beat)

        current_notes = [0 for i in range(127)]
        last_event_time = 0
        index = 0
        current = MidiTrack()

        ticks_per_division = int(midi.ticks_per_beat / (self.division * numerator / denominator))
        for item in self.divisions:
            # xor-ing current notes
            for i in range(127):
                if current_notes[i] != item[i]:
                    if current_notes[i] == 1:
                        current.append(Message('note_off', note=i, velocity=127, time=index - last_event_time))
                        last_event_time = index
                    else:
                        current.append(Message('note_on', note=i, velocity=127, time=index - last_event_time))
                        last_event_time = index
                    current_notes[i] ^= item[i]
                elif current_notes[i] == 1 and index % (self.division * midi.ticks_per_beat) == 0:
                    current.append(Message('note_off', note=i, velocity=127, time=index - last_event_time))
                    last_event_time = index
                    current.append(Message('note_on', note=i, velocity=127, time=0))

            index += ticks_per_division
        last_event_time = index - last_event_time
        if last_event_time != 0:
            current.append(Message('note_off', note=0, velocity=0, time=last_event_time))
        for i in range(127):
            if current_notes[i] == 1:
                current.append(Message('note_off', note=i, velocity=127, time=0))

        midi.tracks.append(current)
        midi.save(name + ".mid")


# ================================================================================================================================
# Реализации интерфейса
# ================================================================================================================================
class CustomTrack1D(CustomTrackInterface):
    """    Класс-обертка для MIDI-трека. Используется для подготовки данных о дорожке MIDI-файла к употреблению нейросетью.
    Attributes:
        division (int): минимальная доля разбиениения, сичтатеся относительно "целой ноты" произведения
        divisions (list): набор подготовленных разбиений файла с частотой division
        numerator (int): Количество долей в такте
        denominator (int): Длительность одной доли
        name (str): Имя трека
        
    """

    def __init__(self, division: int, numerator: int, denominator: int, divisions: list = list, name: str = "default"):
        super().__init__(division, numerator, denominator, divisions, name)

    def get_data_set(self, step: int, sample_length: int, result_length: int = 1) -> [list, list]:
        """
        Метод доступа к подготовленному датасету, при каждом вызове формирует его заново.
        :param step: Длина смещения в долях минимального разбиения
        :param sample_length: Длина единичного семпла для входных данных обучения
        :param result_length: Длина единичного семпла для выходных данных обучения, как правило равна 1
        :return: Пара массивов [X, y], где X входные, а y - выходные
        """
        in_divisions = []
        out_divisions = []

        for start in range(0, len(self.divisions) - sample_length - result_length - 1, step):
            in_divisions.append(self.divisions[start:start + sample_length])
            out_divisions.append(self.divisions[start + sample_length: start + sample_length + result_length])

        return [np.array(in_divisions), np.array(out_divisions)]

    def get_segment_data_set(self, start: int, length: int) -> list:
        """
        Метод доступа к сегменту из датасета
        :param start: Индекс первой доли сегмента в датасете
        :param length: Длина сегмента
        :return: Массив долей заданной длины
        """
        # TODO: Ошибки на размер датасета?
        return self.divisions[start:start + length]
