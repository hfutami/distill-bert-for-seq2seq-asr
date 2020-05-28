import argparse
import configparser
import os
import pickle
import logging
import torch
from torch.nn.utils.rnn import pad_sequence
from pytorch_transformers import BertForMaskedLM, modeling_bert

TOP_K = 8
BATCH_SIZE = 50
LOG_STEP = 10000
SAVE_STEP = 10000

def modified_softmax(x, dim, temp):
    e_max, _ = torch.max((x / temp), dim=dim, keepdim=True)
    e_cared = torch.exp((x / temp) - e_max)
    e_sum = torch.sum(e_cared, dim=dim, keepdim=True)
    return e_cared / e_sum

def get_label(model, device, path, save_path, temp):
    model.eval()
    labels = {}

    with open(path) as f:
        lines = [line.strip() for line in f]

    paths_batch = []
    inputs_batch = []
    seq_lens_batch = []
    mask_pos_batch = []

    logging.info("start processing ...")

    for step, line in enumerate(lines):
        tokens = line.split()
        path = tokens[0]
        mask_pos = int(tokens[1])
        input_ids = list(map(int, tokens[2:]))
        input_tensor = torch.tensor(input_ids)
        seq_len = len(input_ids)

        paths_batch.append(path)
        inputs_batch.append(input_tensor)
        seq_lens_batch.append(seq_len)
        mask_pos_batch.append(mask_pos)

        # make batch then to model
        if (step + 1) % BATCH_SIZE == 0 or (step + 1) == len(lines):
            bsize = len(inputs_batch)

            inputs_pad = pad_sequence(inputs_batch, batch_first=True).to(device)
            max_seq_len = max(seq_lens_batch)
            attn_mask = torch.tensor(
                [[float(l < seq_len) for l in range(max_seq_len)] for seq_len in seq_lens_batch]
            ).to(device)

            with torch.no_grad():
                outputs, = model(inputs_pad, attention_mask=attn_mask, masked_lm_labels=None)
            
            outputs = outputs[:, :, 5:]  # not include <pad>, <unk>, <cls>, <sep>, <mask>
            
            for b in range(bsize):
                out = outputs[b]
                path = paths_batch[b]
                mask_pos = mask_pos_batch[b]

                l_sorted, v_indices = torch.sort(out[mask_pos], descending=True)
                l_topk = l_sorted[:TOP_K]
                l = modified_softmax(l_topk, dim=0, temp=temp)

                if path not in labels:  # convert id.bert to id.asr
                    labels[path] = [(v_indices[:TOP_K].cpu().numpy() + 3, l.cpu().numpy())]
                else:
                    labels[path].append((v_indices[:TOP_K].cpu().numpy() + 3, l.cpu().numpy()))

            paths_batch = []
            inputs_batch = []
            seq_lens_batch = []
            mask_pos_batch = []

        if (step + 1) % LOG_STEP == 0:
            logging.info(f"step: {step + 1} / {len(lines)}")

        if (step + 1) == SAVE_STEP:
            save_tmp = f"{save_path}.tmp"
            with open(save_tmp, "wb") as f:
                pickle.dump(labels, f)
            logging.info(f"pickle is saved to {save_tmp}")

    with open(save_path, "wb") as f:
        pickle.dump(labels, f)
    logging.info(f"pickle is saved to {save_path} (finished)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", type=str)
    parser.add_argument("-model", type=str)
    parser.add_argument("-ctx", type=str)
    parser.add_argument("--temp", type=float, default=3.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.makedirs("../data/log/", exist_ok=True)
    os.makedirs("../data/soft_labels/", exist_ok=True)
    if args.ctx == "utt":
        script_path = f"../data/csj/csj.aps.pathaid.bert.c1.masked"
        log_path = f"../data/log/prep_soft_labels_utt.log"
        save_path = f"../data/soft_labels/bert-utt.labels"
    elif args.ctx == "full":
        script_path = f"../data/csj/csj.aps.pathaid.bert.c256.masked"
        log_path = f"../data/log/prep_soft_labels_full.log"
        save_path = f"../data/soft_labels/bert-full.labels"

    if args.debug:
        logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(filename=log_path,
                            format="%(asctime)s %(message)s",
                            level=logging.DEBUG)
    logging.info(f"script_path: {script_path}")
    logging.info(f"soft labels will be saved to {save_path}")

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.conf)
    vocab_size = int(config["vocab"]["vocab_size"])
    hidden_size = int(config["model"]["hidden_size"])
    num_hidden_layers = int(config["model"]["num_hidden_layers"])
    num_attention_heads = int(config["model"]["num_attention_heads"])
    intermediate_size = int(config["model"]["intermediate_size"])
    max_position_embeddings = int(config["model"]["max_position_embeddings"])

    bertconfig = modeling_bert.BertConfig(vocab_size_or_config_json_file=vocab_size,
                                          hidden_size=hidden_size,
                                          num_hidden_layers=num_hidden_layers,
                                          num_attention_heads=num_attention_heads,
                                          intermediate_size=intermediate_size,
                                          max_position_embeddings=max_position_embeddings)
    model = BertForMaskedLM(config=bertconfig)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    logging.info(f"load model from {args.model}")
    model.to(device)

    get_label(model, device, script_path, save_path, args.temp)
