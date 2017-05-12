from abc import abstractmethod

from mido import MidiFile, MidiTrack, Message, MetaMessage


class OutputController:
    @abstractmethod
    def write_log(self, value):
        pass

    @abstractmethod
    def write_data(self, value):
        pass


class MidiOutput(OutputController):
    def __init__(self, path_to_log: str, division: int, numerator: int, denominator: int):
        """
        Конструктор класса-контроллера вывода данных модели
        :param path_to_log: путь к файлу для логов
        :param division: "разбиение" или минимальная доля произведения 
        :param numerator: количество долей в такте
        :param denominator: размер доли в такте
        # ======================================================================================================================
        # Параметры такта и минимального деления
        # ======================================================================================================================
        # Семплы сообщений
        # <meta message time_signature numerator=4 denominator=4 clocks_per_click=24 notated_32nd_notes_per_beat=8 time=0>
        # note_on channel=0 note=45 velocity=80 time=0
        # note_off channel=0 note=45 velocity=64 time=480
        # ======================================================================================================================
       
        """
        self.log = open(path_to_log, "w")
        self.buffer = []
        self.midi = MidiFile()
        self.division = division
        self.numerator = numerator
        self.denominator = denominator

        if len(self.midi.tracks) == 0:
            self.midi.tracks.append(MidiTrack())
        for index in range(len(self.midi.tracks)):
            track = MidiTrack()
            track.append(
                MetaMessage(
                    'time_signature', numerator=numerator, denominator=denominator, clocks_per_click=24,
                    notated_32nd_notes_per_beat=8))
            self.midi.tracks.append(track)

    def write_log(self, value):
        self.log.write(value)

    def write_data(self, value):
        self.write_log("DATA WRITING OPERATION\n")
        # self.write_log(value)
        self.buffer.append(value)

    def build_midi_file(self, name):
        current_notes = [0 for i in range(127)]
        last_event_time = 0
        index = 0
        ticks_per_division = int(self.midi.ticks_per_beat / self.division * self.denominator)

        for item in self.buffer:
            # xor-ing current notes
            for i in range(127):
                if current_notes[i] != item[i]:
                    if current_notes[i] == 1:
                        self.midi.tracks[0].append(Message('note_off', note=i, velocity=127, time=last_event_time))
                    else:
                        self.midi.tracks[0].append(Message('note_on', note=i, velocity=127, time=last_event_time))
                    last_event_time = index - last_event_time
                    current_notes[i] ^= item[i]
            index += ticks_per_division
        for i in range(127):
            if current_notes[i] == 1:
                self.midi.tracks[0].append(Message('note_off', note=i, velocity=127, time=last_event_time))
        self.log.close()
        self.midi.save(name + '.mid')
