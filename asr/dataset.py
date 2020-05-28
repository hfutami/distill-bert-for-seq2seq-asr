import configparser
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import load_htk, frame_stacking, spec_augment


class SpeechDataset(Dataset):
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        script_path = config["data"]["train_script"]

        self.lmfb_dim = int(config["frontend"]["lmfb_dim"])
        self.specaug = bool(int(config["frontend"]["specaug"]))
        self.num_framestack = int(config["frontend"]["num_framestack"])
        self.vocab_size = int(config["vocab"]["vocab_size"])
        self.distill = bool(float(config["distill"]["distill_weight"]) > 0)
        self.ls_prob = float(config["train"]["ls_prob"])

        if self.specaug:
            max_mask_freq = int(config["frontend"]["max_mask_freq"])
            max_mask_time = int(config["frontend"]["max_mask_time"])
            self.max_mask_freq = max_mask_freq
            self.max_mask_time = max_mask_time

        with open(script_path) as f:
            lines = [line.strip() for line in f.readlines()]
        self.dat = lines

        if self.distill:
            soft_label_path = config["distill"]["soft_label_path"]
            with open(soft_label_path, "rb") as f:
                self.topk_probs_dict = pickle.load(f)

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx):
        xpath, label = self.dat[idx].split(" ", 1)
        x = load_htk(xpath)[:, :self.lmfb_dim]

        if self.specaug:
            x = spec_augment(x, self.lmfb_dim, self.max_mask_freq, self.max_mask_time)
        if self.num_framestack > 1:
            x = frame_stacking(x, self.num_framestack)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        seq_len = x_tensor.shape[0]
        ret = (x_tensor, seq_len,)
        hard_lab = torch.tensor(list(map(int, label.split(" "))))  # (seq_len,)
        lab_len = hard_lab.shape[0]

        if self.distill:
            if xpath in self.topk_probs_dict:
                topk_probs = self.topk_probs_dict[xpath]
            else:
                topk_probs = []

            # (seq_len, vocab_size)
            soft_lab = create_soft_label(topk_probs, lab_len, self.vocab_size, hard_lab, self.ls_prob)
            ret += (hard_lab, soft_lab, lab_len)
        else:
            # without soft_lab
            ret += (hard_lab, lab_len)

        return ret


def collate_fn(batch):
    if len(list(zip(*batch))) == 4:  # distill
        xs, seq_lens, hard_labels, lab_lens = zip(*batch)
        soft_labels = None
    else:
        xs, seq_lens, hard_labels, soft_labels, lab_lens = zip(*batch)
    
    ret = {}
    # (batch, seq_len, dim)
    ret["x_batch"] = pad_sequence(xs, batch_first=True)
    ret["seq_lens"] = torch.tensor(seq_lens)

    # (batch, lab_len)
    ret["hard_labels"] = pad_sequence(hard_labels, batch_first=True)

    if soft_labels is not None:
        # (batch, lab_len - 2, vocab) not include <sos>, <eos>
        ret["soft_labels"] = pad_sequence(soft_labels, batch_first=True)

    ret["lab_lens"] = torch.tensor(lab_lens)

    return ret

def create_soft_label(topk_probs, lab_len, vocab_size, hard_lab, ls_prob=0.9):
    t_probs = torch.zeros(lab_len, vocab_size)
    
    t_probs[0, :] = (1 - ls_prob) / (vocab_size - 1)
    t_probs[0, hard_lab[0]] = 1.0 * ls_prob  # <sos>

    for i, (v_ids, probs) in enumerate(topk_probs):  # loop for lab_len - 2
        t_probs[i + 1, :] = (1 - ls_prob) / (vocab_size - len(v_ids))
        for v_id, prob in zip(v_ids, probs):  # loop for topk
            t_probs[i + 1, v_id] = prob.astype(np.float64) * ls_prob

    t_probs[-1, :] = (1 - ls_prob) / (vocab_size - 1)
    t_probs[-1, hard_lab[-1]] = 1.0 * ls_prob # <eos>

    return t_probs
