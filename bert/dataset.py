import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class LMDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            lines = [line.strip() for line in f.readlines()]
        dat = []
        for line in lines:
            ids = list(map(int, line.split()))
            dat.append(ids)
        
        self.dat = dat

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx):
        input_ids = self.dat[idx]
        input_tensor = torch.tensor(input_ids)
        seq_len = input_tensor.shape[0]
        return input_tensor, seq_len

def collate_fn(batch):
    inputs, seq_lens = zip(*batch)
    ret = {}
    ret["inputs"] = pad_sequence(inputs, batch_first=True)
    ret["seq_lens"] = seq_lens
    max_seq_lens = max(seq_lens)
    ret["attn_mask"] = torch.tensor(
                            [[float(l < seq_len) for l in range(max_seq_lens)] for seq_len in seq_lens]
                       )
    return ret
