import configparser
import random
from operator import itemgetter
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from attention.attention import ContentBasedAttention

import sys
sys.path.append("../")
from utils import map_id_to_bert, convert_probs_to_asr
from vocab import Vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INF_MIN = -1e10


class Decoder(nn.Module):
    def __init__(self, config_path, lm):
        super(Decoder, self).__init__()

        config = configparser.ConfigParser()
        config.read(config_path)

        self.hidden_size = int(config["model"]["hidden_size"])
        self.vocab_size = int(config["vocab"]["vocab_size"])
        self.beam_width = int(config["test"]["beam_width"])
        self.max_seq_len = int(config["test"]["max_seq_len"])

        self.sos_id = int(config["vocab"]["sos_id"])
        self.eos_id = int(config["vocab"]["eos_id"])

        self.ss_prob = float(config["train"]["ss_prob"])
        if self.ss_prob > 0:
            self.ss_type = config["train"]["ss_type"]
            self.ss_start_epoch = int(config["train"]["ss_start_epoch"])
        
        # lm_fusion
        self.lm_fusion = bool(int(config["lm"]["lm_fusion"]))

        if self.lm_fusion:
            self.lm_fusion_type = config["lm"]["lm_fusion_type"]

            if self.lm_fusion_type == "cold":
                # fix LM parameters
                for p in self.lm.parameters():
                    p.requires_grad = False
                
                self.lm.eval()

                self.L_feat = nn.Linear(self.lm.hidden_size, self.hidden_size)
                self.L_gate = nn.Linear(self.hidden_size*2, self.hidden_size)
                self.L_out = nn.Linear(self.hidden_size*2, self.vocab_size)

        self.attn = ContentBasedAttention(config_path)

        # generate
        self.L_sy = nn.Linear(self.hidden_size, self.hidden_size)
        self.L_gy = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # this can be replace with adaptive softmax
        #if self.adaptive_softmax:
        self.L_yy = nn.Linear(self.hidden_size, self.vocab_size)
        #else:
        #    self.L_yy = nn.AdaptiveLogSoftmaxWithLoss(self.hidden_size, self.vocab_size,
        #                                              cutoffs=[self.vocab_size // 25, self.vocab_size // 5],
        #                                              div_value=4.0)

        # recurrency
        # self.L_yr = nn.Linear(self.vocab_size, self.hidden_size * 4)
        self.L_yr = nn.Embedding(self.vocab_size, self.hidden_size * 4)
        self.L_sr = nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.L_gr = nn.Linear(self.hidden_size * 2, self.hidden_size * 4)

        # update by update_epoch
        self.epoch = 0

    def forward(self, h_batch, seq_lens, labels):
        batch_size = h_batch.shape[0]
        frames_len = h_batch.shape[1]
        labels_len = labels.shape[1]

        attn_mask = torch.ones((batch_size, frames_len, 1), device=device, requires_grad=False)
        for b, seq_len in enumerate(seq_lens):
            if b < seq_len:
                attn_mask.data[b, seq_len:] = 0.0

        # for the first time (before <SOS>), generate from this 0-filled hidden_state and cell_state
        s = torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False)
        c = torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False)
        alpha = torch.zeros((batch_size, 1, frames_len), device=device, requires_grad=False)

        preds = torch.zeros((batch_size, labels_len, self.vocab_size), device=device, requires_grad=False)

        if self.lm_fusion and self.lm_fusion_type == "cold":
            lm_hidden = (torch.zeros(self.lm.num_layers,
                                    batch_size,
                                    self.lm.hidden_size, device=device),
                        torch.zeros(self.lm.num_layers,
                                    batch_size,
                                    self.lm.hidden_size, device=device))
            lmout = torch.zeros(batch_size, self.lm.hidden_size, device=device)

        for step in range(labels_len):
            g, alpha = self.attn(s, h_batch, alpha, attn_mask)

            dec_feat = self.L_gy(g) + self.L_sy(s)  # same as Linear(cat(g, s))

            # Cold Fusion
            if self.lm_fusion and self.lm_fusion_type == "cold":
                # generate
                lmfeat = self.L_feat(lmout)
                gate = torch.sigmoid(self.L_gate(torch.cat([dec_feat, lmfeat], dim=-1)))
                gated_lmfeat = gate * lmfeat
                # introduce additional layer ?
                y = self.L_out(torch.tanh(torch.cat([dec_feat, gated_lmfeat], dim=-1)))
            else:
                # generate
                y = self.L_yy(torch.tanh(dec_feat))
            preds[:, step] = y  # (batch, vocab_size)

            # recurrency
            # scheduled sampling
            is_sample = self.ss_prob > 0 and step > 0 and self.epoch >= self.ss_start_epoch and random.random() < self.ss_prob
            # sample or argmax or label sample
            if is_sample and self.ss_type == "hard":
                samps = torch.argmax(preds[:, step - 1], dim=1)  # step > 0
                rec_in = self.L_yr(samps) + self.L_sr(s) + self.L_gr(g)
            else:
                rec_in = self.L_yr(labels[:, step]) + self.L_sr(s) + self.L_gr(g)
            s, c = self._func_lstm(rec_in, c)

            # do not input <sos> to LM
            if self.lm_fusion and self.lm_fusion_type == "cold":
                if step >= 1:
                    lmout, lm_hidden = self.lm.predict(labels[:, step].unsqueeze(1), lm_hidden)
                    lmout = lmout.squeeze(1).detach()  # (batch, lm_hidden_size)

        return preds
    
    def decode(self, h_batch, seq_lens, device, beam_width=None):
        batch_size = h_batch.shape[0]
        assert batch_size == 1

        if beam_width is None:
            beam_width = self.beam_width

        frames_len = h_batch.shape[1]

        # sequence, score, (cell state, hidden state, attention weight)
        beam_paths = [([], 0.0,
                       (torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False),
                        torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False),
                        torch.zeros((batch_size, 1, frames_len), device=device, requires_grad=False)))]

        attn_mask = torch.ones((batch_size, frames_len, 1), device=device, requires_grad=False)
        for b, seq_len in enumerate(seq_lens):
            if b < seq_len:
                attn_mask.data[b, seq_len:] = 0.0
        
        res = []
        for _ in range(self.max_seq_len):
            current_beam_paths = []

            for beam_path in beam_paths:
                cand_seq, cand_score, (c, s, alpha) = beam_path
                g, alpha = self.attn(s, h_batch, alpha, attn_mask)

                y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
                y = log_softmax(y, dim=1)

                y_c = y.clone()
                for _ in range(beam_width):
                    best_idx = y_c.data.argmax(1).item()

                    new_seq = cand_seq + [best_idx]
                    new_score = cand_score + y_c.data[0][best_idx]

                    y_c.data[0][best_idx] = INF_MIN  # this enable to pick up 2nd, 3rd ... best words

                    best_idx_tensor = torch.tensor([best_idx], device=device)
                    rec_in = self.L_yr(best_idx_tensor) + self.L_sr(s) + self.L_gr(g)
                    new_s, new_c = self._func_lstm(rec_in, c)

                    current_beam_paths.append((new_seq, new_score, (new_c, new_s, alpha)))

            # sort by its score
            current_beam_paths_sorted = sorted(current_beam_paths, key=itemgetter(1), reverse=True)

            beam_paths = current_beam_paths_sorted[:beam_width]

            # if top candidate end with <eos>, finish decoding
            if beam_paths[0][0][-1] == self.eos_id:
                for char in beam_paths[0][0]:
                    res.append(char)
                break

        return res

    def decode_tf(self, h_batch, seq_lens, labels, device):
        batch_size = h_batch.shape[0]
        assert batch_size == 1

        frames_len = h_batch.shape[1]

        attn_mask = torch.ones((batch_size, frames_len, 1), device=device, requires_grad=False)
        for b, seq_len in enumerate(seq_lens):
            if b < seq_len:
                attn_mask.data[b, seq_len:] = 0.0
        
        seq = []
        c = torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False)
        s = torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False)
        alpha = torch.zeros((batch_size, 1, frames_len), device=device, requires_grad=False)

        for label in labels:
            g, alpha = self.attn(s, h_batch, alpha, attn_mask)

            y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
            y = log_softmax(y, dim=1)

            y_c = y.clone()
            best_idx = y_c.data.argmax(1).item()

            seq = seq + [best_idx]

            label_tensor = torch.tensor([label], device=device)
            rec_in = self.L_yr(label_tensor) + self.L_sr(s) + self.L_gr(g)
            s, c = self._func_lstm(rec_in, c)

        return seq
    
    def decode_nbest(self, h_batch, seq_lens, num_best, device):
        fflag = False  # decoding is finished or not

        # beam width must match num_best
        beam_width = num_best

        # sequence, score
        l_nbest = []

        batch_size = h_batch.shape[0]
        assert batch_size == 1

        frames_len = h_batch.shape[1]

        # sequence, score, (cell state, hidden state, attention weight)
        beam_paths = [([], 0.0,
                       (torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False),
                        torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False),
                        torch.zeros((batch_size, 1, frames_len), device=device, requires_grad=False)))]

        attn_mask = torch.ones((batch_size, frames_len, 1), device=device, requires_grad=False)
        for b, seq_len in enumerate(seq_lens):
            if b < seq_len:
                attn_mask.data[b, seq_len:] = 0.0
        
        for _ in range(self.max_seq_len):
            current_beam_paths = []

            for beam_path in beam_paths:
                cand_seq, cand_score, (c, s, alpha) = beam_path
                g, alpha = self.attn(s, h_batch, alpha, attn_mask)

                y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
                y = log_softmax(y, dim=1)

                y_c = y.clone()
                for _ in range(beam_width):
                    best_idx = y_c.data.argmax(1).item()

                    new_seq = cand_seq + [best_idx]
                    new_score = cand_score + y_c.data[0][best_idx]

                    y_c.data[0][best_idx] = INF_MIN  # this enable to pick up 2nd, 3rd ... best words

                    best_idx_tensor = torch.tensor([best_idx], device=device)
                    rec_in = self.L_yr(best_idx_tensor) + self.L_sr(s) + self.L_gr(g)
                    new_s, new_c = self._func_lstm(rec_in, c)

                    current_beam_paths.append((new_seq, new_score, (new_c, new_s, alpha)))

            # sort by its score
            current_beam_paths_sorted = sorted(current_beam_paths, key=itemgetter(1), reverse=True)

            cbeam_paths_top = current_beam_paths_sorted[:beam_width]
            len_cbeam_paths = len(cbeam_paths_top)

            new_beam_paths = []

            for idx in range(len_cbeam_paths):
                if cbeam_paths_top[idx][0][-1] == self.eos_id:
                    len_seq = len(cbeam_paths_top[idx][0])

                    # only <sos> and <eos> is not acceptable
                    if len_seq <= 2:
                        continue

                    # seq, score by len_seq
                    l_nbest.append((cbeam_paths_top[idx][0], cbeam_paths_top[idx][1].item()))

                    if len(l_nbest) >= num_best:
                        fflag = True
                        break
                else:
                    new_beam_paths.append(cbeam_paths_top[idx])
            if fflag:
                break
            
            beam_paths = new_beam_paths

        # sort by its score
        sorted_l_nbest = sorted(l_nbest, key=itemgetter(1), reverse=True)

        return sorted_l_nbest
    
    def decode_transfo_fusion(self, h_batch, seq_lens, lmodel, lm_weight, len_weight, device, beam_width=None):
        # debug
        #vocab = Vocab("./data/bpe2k/subword.id.asr")
        #vocab2 = Vocab("./data/bpe2k/subword.id.bert")

        fflag = False  # decoding is finished or not

        batch_size = h_batch.shape[0]
        assert batch_size == 1

        if beam_width is None:
            beam_width = self.beam_width

        # sequence, score
        l_nbest = []

        frames_len = h_batch.shape[1]

        # sequence, score, (cell state, hidden state, attention weight),
        # (lm hidden state, cell state), prev token
        beam_paths = [([],
                       0.0,
                       (torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False),
                        torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False),
                        torch.zeros((batch_size, 1, frames_len), device=device, requires_grad=False)))]

        attn_mask = torch.ones((batch_size, frames_len, 1), device=device, requires_grad=False)
        for b, seq_len in enumerate(seq_lens):
            if b < seq_len:
                attn_mask.data[b, seq_len:] = 0.0
        
        for _ in range(self.max_seq_len):
            current_beam_paths = []

            for beam_path in beam_paths:
                cand_seq, cand_score, (c, s, alpha) = beam_path
                g, alpha = self.attn(s, h_batch, alpha, attn_mask)

                y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
                y = log_softmax(y, dim=1)

                y_c = y.clone()  # (1, vocab_size)

                if len(cand_seq) >= 1:
                    #print("cand_seq:", cand_seq)

                    # convert inputs to id.bert
                    inputs = [map_id_to_bert(c) for c in cand_seq]
                    #print("inputs:", " ".join(vocab2.ids2words(inputs)))
                    inputs = torch.tensor([inputs], device=device)

                    with torch.no_grad():
                        pred, = lmodel(inputs, labels=None) # (1, seq_len, vocab_size)
                    pred = pred[:, -1, :]
                    #print("pred:", pred.shape)

                    # convert pred to id.asr
                    pred = convert_probs_to_asr(pred)
                    
                    score_lm = log_softmax(pred, dim=1).squeeze(1) # (1)
                    #print("score_lm:", score_lm)

                    #print(" ".join(vocab.ids2words(cand_seq)))
                    #print("ASR: ", vocab.id2word(y_c.data.argmax(1).item()))
                    #print("LM: ", vocab.id2word(score_lm.data.argmax(1).item()))
                    # log P(y|X) + lm_weight * log P(y) + len_weight * (len(sequence) + 1)
                    y_c.data = y_c.data + lm_weight * score_lm.data + len_weight
                    # print("fused best: ", vocab.id2word(y_c.data.argmax(1).item()))
                
                for _ in range(beam_width):
                    best_idx = y_c.data.argmax(1).item()

                    new_seq = cand_seq + [best_idx]

                    new_score = cand_score + y_c.data[0][best_idx]

                    y_c.data[0][best_idx] = INF_MIN  # this enable to pick up 2nd, 3rd ... best words

                    best_idx_tensor = torch.tensor([best_idx], device=device)
                    rec_in = self.L_yr(best_idx_tensor) + self.L_sr(s) + self.L_gr(g)
                    new_s, new_c = self._func_lstm(rec_in, c)

                    current_beam_paths.append((new_seq, new_score, (new_c, new_s, alpha)))

            # sort by its score
            current_beam_paths_sorted = sorted(current_beam_paths, key=itemgetter(1), reverse=True)

            cbeam_paths_top = current_beam_paths_sorted[:beam_width]
            len_cbeam_paths = len(cbeam_paths_top)

            new_beam_paths = []

            for idx in range(len_cbeam_paths):
                if cbeam_paths_top[idx][0][-1] == self.eos_id:
                    len_seq = len(cbeam_paths_top[idx][0])

                    # only <sos> and <eos> is not acceptable
                    if len_seq <= 2:
                        continue

                    # seq, score by len_seq
                    l_nbest.append((cbeam_paths_top[idx][0], cbeam_paths_top[idx][1].item()))

                    if len(l_nbest) >= self.beam_width:
                        fflag = True
                        break
                else:
                    new_beam_paths.append(cbeam_paths_top[idx])
            if fflag:
                break
            
            beam_paths = new_beam_paths
        
        if len(l_nbest) < 1:
            return []

        # sort by its score
        sorted_l_nbest = sorted(l_nbest, key=itemgetter(1), reverse=True)

        return sorted_l_nbest[0][0]

    
    def decode_transfo_fusion_nbest(self, h_batch, seq_lens, lmodel, lm_weight, len_weight, num_best, device):
        # beam width must match num_best
        beam_width = num_best

        fflag = False  # decoding is finished or not

        batch_size = h_batch.shape[0]
        assert batch_size == 1

        # sequence, score
        l_nbest = []

        frames_len = h_batch.shape[1]

        # sequence, score, (cell state, hidden state, attention weight),
        # (lm hidden state, cell state), prev token
        beam_paths = [([],
                       0.0,
                       (torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False),
                        torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False),
                        torch.zeros((batch_size, 1, frames_len), device=device, requires_grad=False)))]

        attn_mask = torch.ones((batch_size, frames_len, 1), device=device, requires_grad=False)
        for b, seq_len in enumerate(seq_lens):
            if b < seq_len:
                attn_mask.data[b, seq_len:] = 0.0
        
        for _ in range(self.max_seq_len):
            current_beam_paths = []

            for beam_path in beam_paths:
                cand_seq, cand_score, (c, s, alpha) = beam_path
                g, alpha = self.attn(s, h_batch, alpha, attn_mask)

                y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
                y = log_softmax(y, dim=1)

                y_c = y.clone()  # (1, vocab_size)

                if len(cand_seq) >= 1:
                    #print("cand_seq:", cand_seq)

                    # convert inputs to id.bert
                    inputs = [map_id_to_bert(c) for c in cand_seq]
                    #print("inputs:", " ".join(vocab2.ids2words(inputs)))
                    inputs = torch.tensor([inputs], device=device)

                    with torch.no_grad():
                        pred, = lmodel(inputs, labels=None) # (1, seq_len, vocab_size)
                    pred = pred[:, -1, :]
                    #print("pred:", pred.shape)

                    # convert pred to id.asr
                    pred = convert_probs_to_asr(pred)
                    
                    score_lm = log_softmax(pred, dim=1).squeeze(1) # (1)

                    # log P(y|X) + lm_weight * log P(y) + len_weight * (len(sequence) + 1)
                    y_c.data = y_c.data + lm_weight * score_lm.data + len_weight
                
                for _ in range(beam_width):
                    best_idx = y_c.data.argmax(1).item()

                    new_seq = cand_seq + [best_idx]

                    new_score = cand_score + y_c.data[0][best_idx]

                    y_c.data[0][best_idx] = INF_MIN  # this enable to pick up 2nd, 3rd ... best words

                    best_idx_tensor = torch.tensor([best_idx], device=device)
                    rec_in = self.L_yr(best_idx_tensor) + self.L_sr(s) + self.L_gr(g)
                    new_s, new_c = self._func_lstm(rec_in, c)

                    current_beam_paths.append((new_seq, new_score, (new_c, new_s, alpha)))

            # sort by its score
            current_beam_paths_sorted = sorted(current_beam_paths, key=itemgetter(1), reverse=True)

            cbeam_paths_top = current_beam_paths_sorted[:beam_width]
            len_cbeam_paths = len(cbeam_paths_top)

            new_beam_paths = []

            for idx in range(len_cbeam_paths):
                if cbeam_paths_top[idx][0][-1] == self.eos_id:
                    len_seq = len(cbeam_paths_top[idx][0])

                    # only <sos> and <eos> is not acceptable
                    if len_seq <= 2:
                        continue

                    # seq, score by len_seq
                    l_nbest.append((cbeam_paths_top[idx][0], cbeam_paths_top[idx][1].item()))

                    if len(l_nbest) >= num_best:
                        fflag = True
                        break
                else:
                    new_beam_paths.append(cbeam_paths_top[idx])
            if fflag:
                break
            
            beam_paths = new_beam_paths
        
        if len(l_nbest) < 1:
            return []

        # sort by its score
        sorted_l_nbest = sorted(l_nbest, key=itemgetter(1), reverse=True)

        return sorted_l_nbest

    
    def decode_with_attn(self, h_batch, seq_lens, device):
        batch_size = h_batch.shape[0]
        assert batch_size == 1

        beam_width = 1

        frames_len = h_batch.shape[1]

        # sequence, score, (cell state, hidden state, attention weight)
        beam_paths = [([], 0.0,
                       (torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False),
                        torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False),
                        torch.zeros((batch_size, 1, frames_len), device=device, requires_grad=False)))]

        attn_mask = torch.ones((batch_size, frames_len, 1), device=device, requires_grad=False)
        for b, seq_len in enumerate(seq_lens):
            if b < seq_len:
                attn_mask.data[b, seq_len:] = 0.0
        
        res = []
        alphas = []
        for _ in range(self.max_seq_len):
            current_beam_paths = []

            for beam_path in beam_paths:
                cand_seq, cand_score, (c, s, alpha) = beam_path
                g, alpha = self.attn(s, h_batch, alpha, attn_mask)
                alphas.append(alpha)

                y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
                y = log_softmax(y, dim=1)

                y_c = y.clone()
                for _ in range(beam_width):
                    best_idx = y_c.data.argmax(1).item()

                    new_seq = cand_seq + [best_idx]
                    new_score = cand_score + y_c.data[0][best_idx]

                    y_c.data[0][best_idx] = INF_MIN  # this enable to pick up 2nd, 3rd ... best words

                    best_idx_tensor = torch.tensor([best_idx], device=device)
                    rec_in = self.L_yr(best_idx_tensor) + self.L_sr(s) + self.L_gr(g)
                    new_s, new_c = self._func_lstm(rec_in, c)

                    current_beam_paths.append((new_seq, new_score, (new_c, new_s, alpha)))

            # sort by its score
            current_beam_paths_sorted = sorted(current_beam_paths, key=itemgetter(1), reverse=True)

            beam_paths = current_beam_paths_sorted[:beam_width]

            # if top candidate end with <eos>, finish decoding
            if beam_paths[0][0][-1] == self.eos_id:
                for char in beam_paths[0][0]:
                    res.append(char)
                break

        return res, alphas

    @staticmethod
    def _func_lstm(x, c):
        ingate, forgetgate, cellgate, outgate = x.chunk(4, 1)
        half = 0.5
        ingate = torch.tanh(ingate * half) * half + half
        forgetgate = torch.tanh(forgetgate * half) * half + half
        cellgate = torch.tanh(cellgate)
        outgate = torch.tanh(outgate * half) * half + half
        c_next = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c_next)
        return h, c_next
