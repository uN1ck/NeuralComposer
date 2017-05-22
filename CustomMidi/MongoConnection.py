import glob
import os

from mido import MidiFile
from pymongo import MongoClient

from CustomMidi.CustomTrack import CustomTrack

client = MongoClient()

parsed_midi = client.musician.TrainSet
for filename in glob.glob(os.path.join("../train", '*.mid')):
    midi_file = MidiFile(filename)
    current_track = CustomTrack(division=16, numerator=4, denominator=4)
    current_track.parse_midi_file(midi_file)
    parsed_midi.insert_one(
        {
            "name": filename,
            "division": 16,
            "sizes": [4, 4],

            "data": current_track.divisions
        }
    )
