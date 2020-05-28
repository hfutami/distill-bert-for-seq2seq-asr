import random
from struct import unpack
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_htk(filepath: str) -> np.ndarray:
    fh = open(filepath, "rb")
    spam = fh.read(12)
    _, _, samp_size, _ = unpack(">IIHH", spam)
    veclen = int(samp_size / 4)
    fh.seek(12, 0)
    dat = np.fromfile(fh, dtype=np.float32)
    dat = dat.reshape(int(len(dat) / veclen), veclen)
    dat = dat.byteswap()
    fh.close()
    return dat

def frame_stacking(x, num_framestack):
    newlen = x.shape[0] // num_framestack
    lmfb_dim = x.shape[1]
    x_stacked = x[0:newlen * num_framestack].reshape(newlen, lmfb_dim * num_framestack)

    return x_stacked

def spec_augment(x, lmfb_dim, max_mask_freq, max_mask_time):
    """ frequency masking and time masking
    """
    mask_freq = random.randint(0, max_mask_freq)
    mask_freq_from = random.randint(0, lmfb_dim - mask_freq)
    mask_freq_to = mask_freq_from + mask_freq
    x[:, mask_freq_from:mask_freq_to] = 0.0

    len_t = x.shape[0]
    if len_t > max_mask_time:
        mask_duration = random.randint(0, max_mask_time)
    else:
        mask_duration = random.randint(0, len_t - 1)
    
    mask_time_from = random.randint(0, len_t - mask_duration)
    mask_time_to = mask_time_from + mask_duration
    x[mask_time_from:mask_time_to, :] = 0.0
    
    return x

def to_onehot(label: torch.tensor, num_classes: int) -> torch.tensor:
    """ (batch, seq_len) -> (batch, seq_len, num_classes)
    """
    return torch.eye(num_classes)[label].to(device)

def to_onehot_ls(labels: torch.tensor, num_classes: int, ls_prob: float = 0.9) -> torch.tensor:
    onehot = to_onehot(labels, num_classes)
    onehot_ls = ls_prob * onehot + ((1 - ls_prob) / (num_classes - 1)) * (1 - onehot)

    return onehot_ls

def label_smoothing_loss(preds, labels, lab_lens, vocab_size, ls_prob=0.9):
    batch_size = preds.shape[0]
    loss = 0
    onehot_ls = to_onehot_ls(labels, vocab_size, ls_prob)

    for b in range(batch_size):
        lab_len = lab_lens[b]
        # sum for seq_len, vocab_size
        loss -= torch.sum((log_softmax(preds[b][:lab_len], dim=1) * onehot_ls[b][:lab_len]))

    return loss

def distill_loss(distill_weight, preds, soft_labels, hard_labels, lab_lens, vocab_size, ls_prob=0.9):
    """ loss = distill_weight * loss_soft + (1 - distill_weight) * loss_hard
    """
    batch_size = preds.shape[0]
    onehot_ls = to_onehot_ls(hard_labels, vocab_size, ls_prob)
    loss = 0

    for b in range(batch_size):
        lab_len = lab_lens[b]
        loss_soft = torch.sum(soft_labels[b][:lab_len] * log_softmax(preds[b][:lab_len], dim=1))
        loss_hard = torch.sum(onehot_ls[b][:lab_len] * log_softmax(preds[b][:lab_len], dim=1))
        loss -= distill_weight * loss_soft + (1 - distill_weight) * loss_hard

    return loss

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1: 
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find("LSTM") != -1: 
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param.data)
            if "bias" in name:
                param.data.fill_(0)

    for name, param in m.named_parameters():
        if param.dim() == 1:
            if "decoder.L_gate.bias" in name:
                nn.init.constant_(param, -1.)  # bias

def update_epoch(model, epoch):
    model.decoder.epoch = epoch

def decay_lr(optimizer, epoch, decay_start_epoch, decay_rate):
    """ Decay learning rate per epoch
    """
    if epoch >= decay_start_epoch:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= decay_rate
