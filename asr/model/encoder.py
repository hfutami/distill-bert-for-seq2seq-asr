import configparser
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, config_path):
        super(Encoder, self).__init__()

        config = configparser.ConfigParser()
        config.read(config_path)

        lmfb_dim = int(config["frontend"]["lmfb_dim"])
        num_framestack = int(config["frontend"]["num_framestack"])

        input_size = lmfb_dim * num_framestack

        hidden_size = int(config["model"]["hidden_size"])
        num_layers = int(config["model"]["num_layers"])

        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=0.2, bidirectional=True)

    def forward(self, x_batch, seq_lens):
        x_packed = pack_padded_sequence(x_batch, seq_lens, batch_first=True, enforce_sorted=False)

        h_packed, _ = self.bi_lstm(x_packed)

        h_batch, _ = pad_packed_sequence(h_packed, batch_first=True)

        return h_batch
