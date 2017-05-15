from mido import MidiFile, Message, MidiTrack, MetaMessage

from CustomMidi.CustomTrack import CustomTrack


class CustomMidiFile:
    """
    Миди файл. Класс-контейнер треков, определяет темп, размер и профчие характеристики трека
    """

    def __init__(self, division: int = 8):
        self.tracks = []
        self.numerator = 4
        self.denominator = 4
        self.division = division

    def build_midi_file(self, name: str, numerator: int, denominator: int,
                        notated_32nd_notes_per_beat: int = 8, clocks_per_click: int = 24) -> None:
        """
        Метод  построения midi-файла из набора треков экземпляра класса.
        :param name: Названеи midi-файла, в который будет произведеня сборка
        :param tempo: темп (bpm) 
        :param numerator: Количество долей
        :param denominator: Длительность доли
        :param notated_32nd_notes_per_beat: количество 32х долей на каждую долю = 8
        :param clocks_per_click: количесвто тиков на каждую долю = 24
        :return: None
        """
        midi = MidiFile()

        current_notes = [0 for i in range(127)]
        last_event_time = 0
        index = 0

        for track in self.tracks:
            current = MidiTrack()
            current.append(
                MetaMessage(
                    'time_signature', numerator=numerator, denominator=denominator, clocks_per_click=clocks_per_click,
                    notated_32nd_notes_per_beat=notated_32nd_notes_per_beat))

            ticks_per_division = int(midi.ticks_per_beat / track.division * self.denominator)
            for item in track.divisions:
                # xor-ing current notes
                for i in range(127):
                    if current_notes[i] != item[i]:
                        if current_notes[i] == 1:
                            current.append(Message('note_off', note=i, velocity=127, time=last_event_time))
                        else:
                            current.append(Message('note_on', note=i, velocity=127, time=last_event_time))
                        last_event_time = index - last_event_time
                        current_notes[i] ^= item[i]
                index += ticks_per_division
            last_event_time = index - last_event_time
            for i in range(127):
                if current_notes[i] == 1:
                    current.append(Message('note_off', note=i, velocity=127, time=last_event_time))

            midi.tracks.append(current)
        midi.save(name + '.mid')


def build_custom_midi_file(path: str, division: int) -> CustomMidiFile:
    """
    Метод-строитель класса MidiInput, определяющий датасет для обучения сети, генерирует столкьо датасетов, стколько дорожек в треке

    # ======================================================================================================================
    # Семплы сообщений
    # ======================================================================================================================
    # <meta message time_signature numerator=4 denominator=4 clocks_per_click=24 notated_32nd_notes_per_beat=8 time=0>
    # note_on channel=0 note=45 velocity=80 time=0
    # note_off channel=0 note=45 velocity=64 time=480
    # ======================================================================================================================

    :param path: Путь к файлу mid
    :param division: Минимальная доля разбиения
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
    return new_midi_file
