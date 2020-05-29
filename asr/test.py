import argparse
import os
import configparser
import numpy as np
import torch
from attnmodel.model import AttnModel
from utils import frame_stacking, load_htk

import sys
sys.path.append("../")
from prep.vocab import Vocab, subword_to_word


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", type=str)
    parser.add_argument("-model", type=str)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
        torch.set_num_threads(2)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(args.model, map_location=device)

    config = configparser.ConfigParser()
    config.read(args.conf)

    model = AttnModel(args.conf)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    lmfb_dim = int(config["frontend"]["lmfb_dim"])
    num_framestack = int(config["frontend"]["num_framestack"])
    vocab_path = config["vocab"]["vocab_path"]
    vocab = Vocab(vocab_path=vocab_path)
    test_script = config["data"]["test_script"]

    with open(test_script) as f:
        xpaths = [line.strip() for line in f]

    for xpath in xpaths:
        x = load_htk(xpath)[:, :lmfb_dim]

        if num_framestack > 1:
            x = frame_stacking(x, num_framestack)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        seq_len = x.shape[0]
        seq_lens = torch.tensor([seq_len]).to(device)

        res = model.decode(x_tensor.unsqueeze(0), seq_lens, device)
        res_subword = vocab.ids2words(res)
        res_word = subword_to_word(res_subword)

        print(xpath, " ".join(res_word), flush=True)


if __name__ == "__main__":
    test()
