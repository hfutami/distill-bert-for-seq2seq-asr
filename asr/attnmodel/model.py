import torch
import torch.nn as nn
from attnmodel.encoder import Encoder
from attnmodel.decoder import Decoder


class AttnModel(nn.Module):
    def __init__(self, config_path):
        super(AttnModel, self).__init__()
        self.encoder = Encoder(config_path)
        self.decoder = Decoder(config_path)

    def forward(self, x_batch, seq_lens, labels):
        h_batch = self.encoder(x_batch, seq_lens)
        preds = self.decoder(h_batch, seq_lens, labels)

        return preds

    def decode(self, x, seq_lens, device):
        with torch.no_grad():
            h_batch = self.encoder(x, seq_lens)
            preds = self.decoder.decode(h_batch, seq_lens, device)

        return preds
