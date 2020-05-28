import argparse
from vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument("-text", type=str)
parser.add_argument("-vocab", type=str)
args = parser.parse_args()

vocab = Vocab(args.vocab)

with open(args.text) as f:
    lines = [line.strip() for line in f]
for line in lines:
    words = line.split()
    ids = vocab.words2ids(words)
    ids_str = list(map(str, ids))
    print(" ".join(ids_str))
