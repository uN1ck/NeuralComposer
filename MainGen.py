import argparse

from mido import MidiFile

from Composer.CustomTrack import CustomTrack

parser = argparse.ArgumentParser(
    description='LSTM Composer generator. Need to load trained model and midi-file as a seed of generation.',
    epilog='Easiest way to generate some Du4b!')
parser.add_argument('model_path', metavar='MODEL', type=str, help="Path to the model. Can be absolute or relative.")
parser.add_argument('seed_path', metavar='MIDI', type=str, help="Path to the midi-seed. Can be absolute or relative.")
parser.add_argument('result_path', metavar='RESULT', type=str,
                    help="Where to save generated midi.")
parser.add_argument('division', metavar='BEAT_DIVISION', type=int, help="Divisions per beat of output midi, use 4 as default")

args = parser.parse_args()

print(args)
print("=" * 40 + "\nLSTM Composer\n" + "=" * 40)
# ==========================================================================================================
print("Parsing input midi: " + args.seed_path)
# ----------------------------------------------------------------------------------------------------------
# Parsing input midi file
# ==========================================================================================================
seed_track = CustomTrack(8, 4, 4)
seed_track.parse_midi_file(MidiFile(args.seed_path))
# ==========================================================================================================
print("Loading saved model: " + args.model_path)
# ----------------------------------------------------------------------------------------------------------
# Loading saved model
# ==========================================================================================================
from Composer.Musician import Musician

musician = Musician.load(args.model_path)

# ==========================================================================================================
print("Generating MIDI to: " + args.result_path)
# ----------------------------------------------------------------------------------------------------------
# GENERATION
# ==========================================================================================================
(seed, generated, raw) = musician.generate(seed_track.get_segment_data_set(0, musician.x_size), 32, args.model_path, None)
res_track = CustomTrack(args.division, 4, 4)
res_track.divisions = generated
res_track.build_midi_file(args.result_path, 4, 4)
print("DONE!")
