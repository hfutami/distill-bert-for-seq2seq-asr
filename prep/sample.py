import argparse
import logging


def sample(path, seqlen, shift):
    num_steps = (seqlen // shift)

    with open(path) as f:
        lines = [line.strip() for line in f]
    
    # all tokens in the script
    tokens = []
    for line in lines:
        tokens.extend(line.split())

    for i in range(num_steps):
        st = 0 + i * shift
        while st + seqlen < len(tokens):
            ed = st + seqlen
            print(" ".join(tokens[st:ed]))
            st = ed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--shift", type=int, default=64)
    args = parser.parse_args()
    sample(args.path, args.seqlen, args.shift)
