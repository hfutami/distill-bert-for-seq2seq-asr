import argparse

import sys
sys.path.append("../")
from asr.utils import load_htk

parser = argparse.ArgumentParser()
parser.add_argument("script", type=str)
args = parser.parse_args()

with open(args.script) as f:
    lines = [line.strip() for line in f]

# tuple of (xlen, line_id)
xlens = []

for line_id, line in enumerate(lines):
    xpath = line.split()[0]
    dat = load_htk(xpath)
    xlens.append((dat.shape[0], line_id))

# sort by xlen
xlens_sorted = sorted(xlens)

for xlen, line_id in xlens_sorted:
    print(lines[line_id])
