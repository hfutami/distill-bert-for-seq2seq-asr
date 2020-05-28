import argparse
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("dir", type=str)
parser.add_argument("--step", type=int, default=10000)
args = parser.parse_args()

with open(args.path) as f:
    lines = [line.strip() for line in f]
random.shuffle(lines)

os.makedirs(args.dir, exist_ok=True)

writeline = []
cnt = 0
for i, line in enumerate(lines):
    writeline.append(line + "\n")
    if (i + 1) % args.step == 0:
        save_path = os.path.join(args.dir, f"bccwj.id.train.{cnt}")
        cnt += 1
        with open(save_path, mode="w") as f:
            f.writelines(writeline)
        writeline = []

save_path = os.path.join(args.dir, f"bccwj.id.train.{cnt}")
with open(save_path, mode="w") as f:
    f.writelines(writeline)
