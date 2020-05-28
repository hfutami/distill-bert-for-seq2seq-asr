import torch
import torch.nn as nn
from attention.encoder import Encoder
from attention.decoder import Decoder


class AttnModel(nn.Module):
    def __init__(self, config_path, lm=None):
        super(AttnModel, self).__init__()
        self.encoder = Encoder(config_path)
        self.decoder = Decoder(config_path, lm)

    def forward(self, x_batch, seq_lens, labels):
        h_batch = self.encoder(x_batch, seq_lens)
        preds = self.decoder(h_batch, seq_lens, labels)

        return preds

    def decode(self, x, seq_lens, device, beam_width=None):
        with torch.no_grad():
            h_batch = self.encoder(x, seq_lens)
            preds = self.decoder.decode(h_batch, seq_lens, device, beam_width)

        return preds

    def decode_tf(self, x, seq_lens, labels, device):
        with torch.no_grad():
            h_batch = self.encoder(x, seq_lens)
            preds = self.decoder.decode_tf(h_batch, seq_lens, labels, device)

        return preds
    
    def decode_nbest(self, x, seq_lens, num_best, device):
        with torch.no_grad():
            h_batch = self.encoder(x, seq_lens)
            preds = self.decoder.decode_nbest(h_batch, seq_lens, num_best, device)

        return preds

    def decode_fusion(self, x, seq_lens, lmodel, lm_num_layers, lm_hidden_size, lm_weight, len_weight, device):
        with torch.no_grad():
            h_batch = self.encoder(x, seq_lens)
            preds = self.decoder.decode_fusion(h_batch, seq_lens, lmodel, lm_num_layers, lm_hidden_size, lm_weight, len_weight, device)
        
        return preds

    def decode_transfo_fusion(self, x, seq_lens, lmodel, lm_weight, len_weight, device, beam_width=None):
        with torch.no_grad():
            h_batch = self.encoder(x, seq_lens)
            preds = self.decoder.decode_transfo_fusion(h_batch, seq_lens, lmodel, lm_weight, len_weight, device, beam_width)
        
        return preds
    
    def decode_transfo_fusion_nbest(self, x, seq_lens, lmodel, lm_weight, len_weight, num_best, device):
        with torch.no_grad():
            h_batch = self.encoder(x, seq_lens)
            preds = self.decoder.decode_transfo_fusion_nbest(h_batch, seq_lens, lmodel, lm_weight, len_weight, num_best, device)
        
        return preds

    def decode_with_attn(self, x, seq_lens, device):
        with torch.no_grad():
            h_batch = self.encoder(x, seq_lens)
            preds, alphas = self.decoder.decode_with_attn(h_batch, seq_lens, device)

        return preds, alphas
