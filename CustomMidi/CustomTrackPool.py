import glob
import os

from mido import MidiFile

from CustomMidi.CustomTrack import CustomTrack


class CustomTrackPool:
    def __init__(self, path_to_data_pool):
        self.data_pool = []
        self._index = 0
        self.division = 8

        if path_to_data_pool is not None:
            for filename in glob.glob(os.path.join(path_to_data_pool, '*.mid')):
                midi_file = MidiFile(filename)
                # TODO: Make builder to this
                # ==================================================================
                current_track = CustomTrack(division=8, numerator=4, denominator=4)
                current_track.parse_midi_file(midi_file)
                # ==================================================================

                self.data_pool.append(current_track)

    def __iter__(self):
        return self.data_pool

    def build_midi_files(self, path):
        for i in range(len(self.data_pool)):
            self.data_pool[i].build_midi_file(path + "\\" + str(i), 4, 4)
