import glob
import os
from abc import abstractmethod

from mido import MidiFile
from pymongo import MongoClient

from CustomMidi.CustomTrack import CustomTrack


class CustomTrackPoolInterface:
    @abstractmethod
    def get_data_pool(self):
        pass

    @abstractmethod
    def put_track(self, value: CustomTrack, name: str):
        pass


class CustomTrackPool(CustomTrackPoolInterface):
    def put_track(self, value: CustomTrack, name: str):
        self.data_pool.append(value)

    def get_data_pool(self):
        return self.data_pool

    def __init__(self, path_to_data_pool, division: int):
        self.data_pool = []
        self._index = 0
        self.division = 8

        if path_to_data_pool is not None:
            for filename in glob.glob(os.path.join(path_to_data_pool, '*.mid')):
                midi_file = MidiFile(filename)
                # TODO: Make builder to this
                # ==================================================================
                current_track = CustomTrack(division=division, numerator=4, denominator=4)
                current_track.parse_midi_file(midi_file)
                # ==================================================================

                self.data_pool.append(current_track)

    def __iter__(self):
        return self.data_pool

    def build_midi_files(self, path):
        for i in range(len(self.data_pool)):
            self.data_pool[i].build_midi_file(path + "\\" + str(i), 4, 4)


class MongoDBTrackPool(CustomTrackPoolInterface):
    def __init__(self):
        client = MongoClient()
        self.data_set = client.musician.TrainSet
        self.data_result = client.musician.ResultSet

    def put_track(self, value: CustomTrack, name: str):
        self.data_result.insert_one(
            {
                "name": name,
                "division": value.division,
                "sizes": [value.numerator, value.denominator],
                "data": value.divisions
            }
        )

    def get_data_pool(self):
        return self.data_set.aggregate({"$project": {"name": 1, "data": 1}})
