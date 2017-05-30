import argparse

parser = argparse.ArgumentParser(
    description='LSTM Composer generator. Need to load trained model and midi-file as a seed of generation.',
    epilog='Easiest way to generate some Du4b!')
parser.add_argument('model_path', metavar='MODEL', type=str, help="Path to the model. Can be absolute or relative.")
parser.add_argument('seed_path', metavar='MIDI', type=str, help="Path to the midi-seed. Can be absolute or relative.")
parser.add_argument('result_path', metavar='RESULT', type=str,
                    help="Where to save generated midi.")
args = parser.parse_args()

print(args)
print("=" * 40 + "\nLSTM Composer v0.2\n" + "=" * 40)
print("Parsing input midi: " + args.seed_path)
print("Loading saved model: " + args.model_path)

from keras.models import load_model

model = load_model(args.model_path)
