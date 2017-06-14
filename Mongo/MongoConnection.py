import glob
import os

from mido import MidiFile, merge_tracks
from pymongo import MongoClient

from Composer.CustomTrack import CustomTrack1D


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
        merged = merge_tracks(midi.tracks)

        for track in [merged]:
            if len(track) < 128:
                continue

            print(str(track))
            track_interpretation = []
            current = [0 for i in range(127)]
            add_to_parsing = True

            for message in track:
                if message.time > 0:
                    for i in range(int(message.time / ticks_per_division)):
                        track_interpretation.append(list(current))

                if message.type == 'note_on':
                    current[message.note] = 1
                if message.type == 'note_off':
                    current[message.note] = 0
                global_time += message.time

            if add_to_parsing and len(track_interpretation) > 64:
                result_interpretation.append(track_interpretation)

        result = ParsedMidi(name=path, division=32, numerator=numerator, denominator=denominator,
                            track_count=len(result_interpretation),
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
            current = CustomTrack1D(division=item["division"] // 2,
                                    name=item["name"],
                                    numerator=item["sizes"][0],
                                    denominator=item["sizes"][1],
                                    divisions=item["data"])

            current.build_midi_file(name=str(collection_name + "/" + str(index)), numerator=4, denominator=4)
            index += 1

    @staticmethod
    def statistics_intervals(collection_name):
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

    @staticmethod
    def statistics_notes_octave(collection_name):
        client = MongoClient()
        data_collection = client.musician[collection_name]
        data_list = list(data_collection.find({}))

        notes = [0 for i in range(12)]
        summon_beats = 0

        for data_item in data_list:
            for item in data_item['data']:
                summon_beats += 1
                notes_sounds = [0 for i in range(12)]
                for i in range(127):
                    notes_sounds[i % 12] += item[i] == 1
                index = 0
                for i in range(11):
                    index += notes_sounds[i] > 0
                notes[index] += 1
            print(data_item['name'])

        for item in range(len(notes)):
            notes[item] /= float(summon_beats)

        return notes

    @staticmethod
    def statistics_notes_total(collection_name):
        client = MongoClient()
        data_collection = client.musician[collection_name]
        data_list = list(data_collection.find({}))

        notes = [0 for i in range(128)]
        summon_beats = 0

        for data_item in data_list:
            for item in data_item['data']:
                summon_beats += 1
                notes_count = 0
                for i in range(127):
                    notes_count += item[i] == 1
                notes[notes_count] += 1
            print(data_item['name'])

        for item in range(len(notes)):
            notes[item] /= float(summon_beats)

        return notes

    @staticmethod
    def render(collection_name, delta=4):

        client = MongoClient()
        data_collection = client.musician[collection_name]
        data_list = list(data_collection.find({}))
        try:
            os.mkdir(collection_name)
        except:
            print("Directory Exists!")

        notes = [0 for i in range(128)]
        summon_beats = 0

        cn = 0
        for data_item in data_list:
            rendered = []

            for item in data_item['raw']:
                maxed = list(set(sorted(item, reverse=True)))
                if len(maxed) > 2:
                    delta = abs(maxed[0] - maxed[2])
                else:
                    delta = abs(maxed[0] - maxed[1])

                max_val = maxed[0]
                buffer = []
                for i in range(len(item)):
                    buffer.append(1 if abs(max_val - item[i]) <= delta else 0)
                rendered.append(buffer)
            result = CustomTrack1D(4, 4, 4, divisions=rendered)
            result.build_midi_file(name=str(collection_name + "\\" + str(cn)), numerator=4, denominator=4)

            cn += 1
            print(cn)


# ==================================================================
# РАНТАЙМ!1
# ==================================================================

# MongoConnection.from_db("3_CNNEDE_ADAM_SQUARED_8")
# MongoConnection.from_db("3_CNNEDE_ADAM_SQUARED_16")
# MongoConnection.from_db("3_CNNEDE_ADAM_SQUARED_32")
# MongoConnection.from_db("3_CNNEDE_ADAM_SQUARED_64")
# MongoConnection.from_db("TrainSet")

from plotly.offline import plot
import plotly.graph_objs as go

trace0 = go.Scatter(x=[i + 1 for i in range(127)], y=MongoConnection.statistics_intervals("TrainSet"), name="Origin")
trace1 = go.Scatter(x=[i + 1 for i in range(127)], y=MongoConnection.statistics_intervals("3_CNNEDE_RMS_SQUARED_8"), name="RMS 8")
trace2 = go.Scatter(x=[i + 1 for i in range(127)], y=MongoConnection.statistics_intervals("3_CNNEDE_RMS_SQUARED_16"),
                    name="RMS 16")
trace3 = go.Scatter(x=[i + 1 for i in range(127)], y=MongoConnection.statistics_intervals("3_CNNEDE_RMS_SQUARED_32"),
                    name="RMS 32")
trace4 = go.Scatter(x=[i + 1 for i in range(127)], y=MongoConnection.statistics_intervals("3_CNNEDE_RMS_SQUARED_64"),
                    name="RMS 64")

data = [trace0, trace1, trace2, trace3, trace4]

plot(data, filename='basic-bar')
