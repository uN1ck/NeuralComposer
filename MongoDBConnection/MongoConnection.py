import glob
import os

from mido import MidiFile
from pymongo import MongoClient

from CustomMidi.CustomTrack import CustomTrack


class parsed_midi:
    def __init__(self, name: str, division: int, numerator: int, denominator: int, track_count: int = 1,
                 divisions: list = list()):
        self.name = name
        self.division = division
        self.divisions = divisions
        self.numerator = numerator
        self.denominator = denominator

    def insert_item(self, collection):
        for track in range(len(self.divisions)):
            if len(self.divisions[track]) > 0:
                res = {
                    "name": self.name,
                    "division": self.division,
                    "sizes": [self.numerator, self.denominator],
                    "is_merged": False,
                    "data": self.divisions[track],
                }
                collection.insert_one(res)


def to_db(collection_name):
    client = MongoClient()
    parsed_midi_collection = client.musician[collection_name]
    for filename in glob.glob(os.path.join("../train_im", '*.mid')):
        midi_file = parse_midi_file(filename, 4)  # MidiFile(filename)
        midi_file.insert_item(parsed_midi_collection)


def parse_midi_file(path: str, division: int) -> parsed_midi:
    midi = MidiFile(path)
    numerator = 4
    denominator = 4

    for track in midi.tracks:
        for message in track:
            if message.is_meta and message.type == 'time_signature':
                numerator = message.numerator
                denominator = message.denominator

    ticks_per_division = int(midi.ticks_per_beat / (division * numerator / denominator))
    global_time = 0

    result_interpretation = []
    for track in midi.tracks:
        print(str(track))
        track_interpretation = []
        current = [0 for i in range(127)]

        add_to_parsing = True
        for message in track:

            if not message.is_meta:
                if message.channel == 9:
                    add_to_parsing = False
                    break
                elif message.type == "program_change" and 9 <= message.program <= 16 and message.program <= 112:
                    add_to_parsing = False
                    break

            if message.time > 0:
                for i in range(int(message.time / ticks_per_division)):
                    track_interpretation.append(list(current))

            if message.type == 'note_on':
                current[message.note] = 1
            if message.type == 'note_off':
                current[message.note] = 0
            global_time += message.time

        if add_to_parsing:
            result_interpretation.append(track_interpretation)

    result = parsed_midi(name=path, division=32, numerator=numerator, denominator=denominator, track_count=len(midi.tracks),
                         divisions=result_interpretation)
    return result


def from_db(collection_name):
    client = MongoClient()
    data_collection = client.musician[collection_name]
    index = 0
    for item in list(data_collection.find({})):
        print("YEP!")
        current = CustomTrack(division=item["division"],
                              name=item["name"],
                              numerator=item["sizes"][0],
                              denominator=item["sizes"][1],
                              divisions=item["data"])
        current.build_midi_file(name=str(index), numerator=4, denominator=4)
        index += 1


# ==================================================================
# РАНТАЙМ!
# ==================================================================

from_db("ResultSet")
