import configparser
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import load_htk
from frontend import frame_stacking, spec_augment

eos_id = 1  # later changed
with_teacher = False


class SpeechDataset(Dataset):
    def __init__(self, config_path, mode="train"):
        config = configparser.ConfigParser()
        config.read(config_path)
        
        if mode == "train":
            script_path = config["data"]["train_script"]
        elif mode == "valid":
            script_path = config["data"]["valid_script"]

        lmfb_dim = int(config["frontend"]["lmfb_dim"])
        specaug = bool(int(config["frontend"]["specaug"]))
        num_framestack = int(config["frontend"]["num_framestack"])
        vocab_size = int(config["vocab"]["vocab_size"])
        global eos_id
        eos_id = int(config["vocab"]["eos_id"])
        global with_teacher
        with_teacher = bool(float(config["distill"]["distill_weight"]) > 0)
        self.ls_prob = float(config["train"]["ls_prob"])
        self.ls_type = config["train"]["ls_type"]

        self.lmfb_dim = lmfb_dim
        self.specaug = specaug
        self.num_framestack = num_framestack
        self.vocab_size = vocab_size

        if specaug:
            max_mask_freq = int(config["frontend"]["max_mask_freq"])
            max_mask_time = int(config["frontend"]["max_mask_time"])
            self.max_mask_freq = max_mask_freq
            self.max_mask_time = max_mask_time

        with open(script_path) as f:
            lines = [line.strip() for line in f.readlines()]

        self.dat = lines

        if with_teacher:
            self.causal = bool(int(config["distill"]["causal"]))
            pickle_path = config["distill"]["pickle_path"]

            with open(pickle_path, "rb") as f:
                self.topk_probs_dict = pickle.load(f)

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx):
        xpath, label = self.dat[idx].split(" ", 1)

        if os.path.splitext(xpath)[-1] == ".htk":
            x = load_htk(xpath)[:, :self.lmfb_dim]
        elif os.path.splitext(xpath)[-1] == ".npy":
            # print("Load npy")
            x = np.load(xpath)

        if self.specaug:
            x = spec_augment(x, self.lmfb_dim, self.max_mask_freq, self.max_mask_time)

        if self.num_framestack > 1:
            x = frame_stacking(x, self.num_framestack)

        x_tensor = torch.tensor(x, dtype=torch.float32)

        seq_len = x_tensor.shape[0]

        ret = (x_tensor, seq_len,)

        hard_lab = torch.tensor(list(map(int, label.split(" "))))  # (seq_len,)
        lab_len = hard_lab.shape[0]

        if with_teacher:
            if xpath in self.topk_probs_dict:
                topk_probs = self.topk_probs_dict[xpath]
            else:
                topk_probs = []

            # (seq_len, vocab_size)
            soft_lab = make_soft_label(topk_probs, lab_len, self.vocab_size, self.causal, hard_lab,
                                       self.ls_type, self.ls_prob)
            
            ret += (hard_lab, soft_lab, lab_len)
        else:
            # without soft_lab
            ret += (hard_lab, lab_len)

        return ret


def collate_fn_train(batch):
    # with teacher or not
    if len(list(zip(*batch))) == 4:
        xs, seq_lens, hard_labels, lab_lens = zip(*batch)
        soft_labels = None
    else:
        xs, seq_lens, hard_labels, soft_labels, lab_lens = zip(*batch)
    
    ret = {}
    # (batch, seq_len, dim)
    ret["x_batch"] = pad_sequence(xs, batch_first=True)
    ret["seq_lens"] = torch.tensor(seq_lens)

    # (batch, lab_len)
    ret["hard_labels"] = pad_sequence(hard_labels, batch_first=True, padding_value=eos_id)

    if soft_labels is not None:
        # (batch, lab_len - 2, vocab) not include <sos>, <eos>
        ret["soft_labels"] = pad_sequence(soft_labels, batch_first=True, padding_value=eos_id)

    ret["lab_lens"] = torch.tensor(lab_lens)

    return ret

def make_soft_label(topk_probs, lab_len, vocab_size, causal, hard_lab, ls_type="uniform", ls_prob=0.9):
    # TODO: adapt to other smoothing method: temporal smoothing
    # For now, unigram smoothing is only available
    if ls_type != "uniform":
        ls_prob = 1.0

    t_probs = torch.zeros(lab_len, vocab_size)

    # because topk_probs not include <sos>, <eos>, first(0) and last(label - 1) is set to hard label
    # in unidirectional lm(uni=True), second(1) is also set to hard label
    t_probs[0, :] = (1 - ls_prob) / (vocab_size - 1)
    t_probs[0, hard_lab[0]] = 1.0 * ls_prob  # <sos>

    if causal:
        t_probs[1, :] = (1 - ls_prob) / (vocab_size - 1)
        t_probs[1, hard_lab[1]] = 1.0 * ls_prob
        for i, (v_ids, probs) in enumerate(topk_probs):  # loop for lab_len - 3
            t_probs[i + 2, :] = (1 - ls_prob) / (vocab_size - len(v_ids))
            for v_id, prob in zip(v_ids, probs):  # loop for topk
                t_probs[i + 2, v_id] = prob.astype(np.float64) * ls_prob
    else:
        for i, (v_ids, probs) in enumerate(topk_probs):  # loop for lab_len - 2
            t_probs[i + 1, :] = (1 - ls_prob) / (vocab_size - len(v_ids))
            for v_id, prob in zip(v_ids, probs):  # loop for topk
                t_probs[i + 1, v_id] = prob.astype(np.float64) * ls_prob

    t_probs[-1, :] = (1 - ls_prob) / (vocab_size - 1)
    t_probs[-1, hard_lab[-1]] = 1.0 * ls_prob # <eos>

    return t_probs
