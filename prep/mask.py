import argparse

parser = argparse.ArgumentParser()
parser.add_argument("script_path", type=str)
parser.add_argument("--mask_id", type=int, default=4)
parser.add_argument("--no_ctx", action="store_true")
args = parser.parse_args()

with open(args.script_path) as f:
    lines = [line.strip() for line in f]

for line in lines:
    tokens = line.split()
    path = tokens[0]
    if args.no_ctx:
        start_mask = 0
        ids = tokens[1:]
        end_mask = len(ids)
    else:
        start_mask = int(tokens[1])
        end_mask = int(tokens[2])
        ids = tokens[3:]

    for index in range(start_mask, end_mask):
        ids_masked = ids.copy()
        ids_masked[index] = str(args.mask_id)
        print(path, index, " ".join(ids_masked))
