import argparse
from vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument("-text", type=str)
parser.add_argument("-vocab", type=str)
parser.add_argument("--addsp", action="store_true")
parser.add_argument("--eos_id", type=int, default=1)
parser.add_argument("--sos_id", type=int, default=2)
args = parser.parse_args()

vocab = Vocab(args.vocab)

with open(args.text) as f:
    lines = [line.strip() for line in f]
for line in lines:
    words = line.split()
    ids = vocab.words2ids(words)
    if args.addsp:
        ids = [args.sos_id] + ids + [args.eos_id]    
    ids_str = list(map(str, ids))
    print(" ".join(ids_str))
