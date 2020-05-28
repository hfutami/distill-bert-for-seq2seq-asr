import argparse

parser = argparse.ArgumentParser()
parser.add_argument("script_path", type=str)
parser.add_argument("-num_ctx", type=int)
args = parser.parse_args()

with open(args.script_path) as f:
    lines = [line.strip() for line in f]

all = []
path2idx = {}
for line in lines:
    tokens = line.split()
    path2idx[tokens[0]] = len(all)
    all.extend(list(map(int, tokens[1:])))

for line in lines:
    tokens = line.split()
    path = tokens[0]
    ids = list(map(int, tokens[1:]))
    s_id = path2idx[path]
    if len(ids) >= args.num_ctx:
        s_id = path2idx[path]
        l_id = s_id
        r_id = s_id + len(ids)
        res = all[l_id : r_id]
        res = " ".join(list(map(str, res)))
        print(f"{path} 0 {len(ids)} {res}")
        continue
    num_l = (args.num_ctx - len(ids)) // 2
    num_r = (args.num_ctx - len(ids)) - ((args.num_ctx - len(ids)) // 2)

    l_id = max(s_id - num_l, 0)
    r_id = min(s_id + len(ids) + num_r, len(all))
    start_mask = (s_id - l_id)
    end_mask = len(ids) + (s_id - l_id)
    res = all[l_id : r_id]
    res = " ".join(list(map(str, res)))
    print(f"{path} {start_mask} {end_mask} {res}")
