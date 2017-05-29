import glob
import os

from mido import MidiFile
from pymongo import MongoClient

from CustomMidi.CustomTrack import CustomTrack


class ParsedMidi:
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


class MongoConnection:
    @staticmethod
    def to_db(collection_name, folder_path="../train"):
        client = MongoClient()
        parsed_midi_collection = client.musician[collection_name]
        for filename in glob.glob(os.path.join(folder_path, '*.mid')):
            midi_file = MongoConnection.parse_midi_file(filename, 4)  # MidiFile(filename)
            midi_file.insert_item(parsed_midi_collection)

    @staticmethod
    def parse_midi_file(path: str, division: int) -> ParsedMidi:
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

        result = ParsedMidi(name=path, division=32, numerator=numerator, denominator=denominator, track_count=len(midi.tracks),
                            divisions=result_interpretation)
        return result

    @staticmethod
    def from_db(collection_name):
        client = MongoClient()
        data_collection = client.musician[collection_name]
        index = 0
        os.mkdir(collection_name)

        for item in list(data_collection.find({})):
            print("YEP!")
            current = CustomTrack(division=item["division"],
                                  name=item["name"],
                                  numerator=item["sizes"][0],
                                  denominator=item["sizes"][1],
                                  divisions=item["data"])

            current.build_midi_file(name=str(collection_name + "/" + str(index)), numerator=4, denominator=4)
            index += 1

    @staticmethod
    def statistics(collection_name):
        client = MongoClient()
        data_collection = client.musician[collection_name]
        data_list = list(data_collection.find({}))

        intervals = [0 for i in range(24)]
        intervals_count_total = 0
        for data_item in data_list:
            current_notes = [0 for i in range(127)]

            last_event_time = 0
            index = 0
            for item in data_item['data']:
                # xor-ing current notes
                for i in range(127):
                    if current_notes[i] == 0 and item[i] == 1:

                        interval = 1
                        while interval <= 24 and i + interval < 127:
                            if current_notes[i + interval] == 1:
                                intervals_count_total += 1
                                intervals[interval - 1] += 1
                            interval += 1
                        interval = 1
                        while interval <= 24 and i - interval >= 0:
                            if current_notes[i - interval] == 1:
                                intervals_count_total += 1
                                intervals[interval - 1] += 1
                            interval += 1

                        last_event_time = index - last_event_time
                        current_notes[i] ^= item[i]
                index += 1

                last_event_time = index - last_event_time
            print(data_item['name'])

        print("Total: " + str(intervals_count_total))
        for item in range(len(intervals)):
            intervals[item] /= float(intervals_count_total)

        return intervals


# ==================================================================
# РАНТАЙМ!
# ==================================================================

MongoConnection.to_db("TrainSet")

# from plotly.offline import plot
# import plotly.graph_objs as go
#
# trace1 = go.Bar(x=[i + 1 for i in range(24)], y=MongoConnection.statistics("TrainSet"), name="Train")
# trace2 = go.Bar(x=[i + 1 for i in range(24)], y=MongoConnection.statistics("RMS_ABSOLUTE_16_4"), name="RMS_4")
# trace3 = go.Bar(x=[i + 1 for i in range(24)], y=MongoConnection.statistics("ADAM_ABSOLUTE_16_4"), name="Adam_4")
# trace4 = go.Bar(x=[i + 1 for i in range(24)], y=MongoConnection.statistics("RMS_ABSOLUTE_16_8"), name="RMS_8")
# trace5 = go.Bar(x=[i + 1 for i in range(24)], y=MongoConnection.statistics("ADAM_ABSOLUTE_16_8"), name="Adam_8")
#
# data = [trace1, trace2, trace3, trace4, trace5]
#
# plot(data, filename='basic-bar')
