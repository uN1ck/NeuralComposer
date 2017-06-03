import glob
import os
from abc import abstractmethod

from mido import MidiFile
from pymongo import MongoClient

from Composer.CustomTrack import CustomTrack


# ================================================================================================================================
# Интерфейс для пула треков
# ================================================================================================================================
class CustomTrackPoolInterface:
    def __init__(self):
        self.data_set = []
        self._index = 0

    @abstractmethod
    def put_track(self, value: CustomTrack, name: str):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


# ================================================================================================================================
# Реализации интерфейса
# ================================================================================================================================
class FileTrackPool(CustomTrackPoolInterface):
    def __next__(self):
        if self._index < len(self.data_set):
            self._index += 1
            return self.data_set[self._index - 1]
        else:
            raise StopIteration

    def put_track(self, value: CustomTrack, name: str):
        self.data_set.append(value)

    def __init__(self, path_to_data_pool, division: int):
        super().__init__()
        if path_to_data_pool is not None:
            for filename in glob.glob(os.path.join(path_to_data_pool, '*.mid')):
                midi_file = MidiFile(filename)
                # TODO: Make builder to this
                # ==================================================================
                current_track = CustomTrack(division=division, numerator=4, denominator=4)
                current_track.parse_midi_file(midi_file)
                # ==================================================================

                self.data_set.append(current_track)

    def __iter__(self):
        return self


class MongoDBTrackPool(CustomTrackPoolInterface):
    def __init__(self, collection_name: str):
        super().__init__()
        client = MongoClient()
        self.data_set = client.musician[collection_name]
        self._count = self.data_set.find({}).count()

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= self._count:
            raise StopIteration
        else:
            self._index += 1
            item = self.data_set.find({})[self._index - 1]
            # TODO: Конструктор из модели бд намутить
            result = CustomTrack(division=item['division'],
                                 numerator=item['sizes'][0],
                                 denominator=item['sizes'][1],
                                 divisions=item["data"],
                                 name=item["name"])
            return result

    def put_track(self, value: CustomTrack, raw: list = None):
        self.data_set.insert_one(
            {
                "name": value.name,
                "division": value.division,
                "sizes": [value.numerator, value.denominator],
                "data": value.divisions,
                "raw": raw,
                "trackPoolId": hash(self)
            }
        )
