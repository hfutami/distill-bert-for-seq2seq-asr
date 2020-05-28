import argparse
import configparser
from datetime import datetime
import logging
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from attention.attn_model import AttnModel
from rnnlm import RNNLM
from dataset import SpeechDataset, collate_fn_train
from loss import distill_loss, label_smoothing_loss
from utils import update_epoch, decay_lr, init_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(model, optimizer, data, vocab_size, ls_type, ls_prob, distill_weight):
    x_batch = data["x_batch"].to(device)
    seq_lens = data["seq_lens"].to(device)
    hard_labels = data["hard_labels"].to(device)
    lab_lens = data["lab_lens"].to(device)

    optimizer.zero_grad()

    preds = model(x_batch, seq_lens, hard_labels)

    if distill_weight > 0:
        soft_labels = data["soft_labels"].to(device)
        loss = distill_loss(distill_weight,
                            preds,
                            soft_labels,
                            hard_labels,
                            lab_lens,
                            vocab_size,
                            ls_type,
                            ls_prob)
    else:
        loss = label_smoothing_loss(preds,
                                    hard_labels,
                                    lab_lens,
                                    vocab_size,
                                    ls_type,
                                    ls_prob)

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    return loss.item()


def val_step(model, data, vocab_size, ls_type, ls_prob, distill_weight):
    x_batch = data["x_batch"].to(device)
    seq_lens = data["seq_lens"].to(device)
    hard_labels = data["hard_labels"].to(device)
    lab_lens = data["lab_lens"].to(device)

    with torch.no_grad():
        preds = model(x_batch, seq_lens, hard_labels)

    if distill_weight > 0:
        soft_labels = data["soft_labels"].to(device)
        loss = distill_loss(distill_weight,
                            preds,
                            soft_labels,
                            hard_labels,
                            lab_lens,
                            vocab_size,
                            ls_type,
                            ls_prob)
    else:
        loss = label_smoothing_loss(preds,
                                    hard_labels,
                                    lab_lens,
                                    vocab_size,
                                    ls_type,
                                    ls_prob)
    
    torch.cuda.empty_cache()

    return loss.item()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="params.conf")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--optim_path", type=str, default=None)
    parser.add_argument("--st_epoch", type=int, default=0)
    args = parser.parse_args()

    config_path = args.config_path
    debug = args.debug

    dt_now = datetime.now()
    dt_str = dt_now.strftime("%m%d%H%M%S")

    # load configs
    config = configparser.ConfigParser()
    config.read(config_path)

    log_dir = config["log"]["log_dir"]
    if config.has_option("log", "log_path"):
        log_path = log_dir + config["log"]["log_path"]
    else:
        log_path = log_dir + "train_attn_{}.log".format(dt_str)
    log_step = int(config["log"]["log_step"])

    save_dir = config["save"]["save_dir"]
    if config.has_option("save", "save_prefix"):
        save_format = save_dir + config["save"]["save_prefix"] + ".network.epoch{}"
        optimizer_save_format = save_dir + config["save"]["save_prefix"] + ".optimizer.epoch{}"
    else:
        save_format = save_dir + "attention{}".format(dt_str) + ".network.epoch{}"
        optimizer_save_format = save_dir + "attention{}".format(dt_str) + ".optimizer.epoch{}"
    save_step = int(config["save"]["save_step"])

    num_epochs = int(config["train"]["num_epochs"])
    batch_size = int(config["train"]["batch_size"])
    decay_start_epoch = int(config["train"]["decay_start_epoch"])
    decay_rate = float(config["train"]["decay_rate"])
    vocab_size = int(config["vocab"]["vocab_size"])
    ls_type = config["train"]["ls_type"]
    ls_prob = float(config["train"]["ls_prob"])

    distill_weight = float(config["distill"]["distill_weight"])
    distill_grad = bool(int(config["distill"]["distill_grad"]))
    if distill_grad:
        distill_weight_fin = distill_weight
        distill_weight = 0
        distill_start_epoch = int(config["distill"]["distill_start_epoch"])
        distill_increase_range = int(config["distill"]["distill_increase_range"])
    
    lm_fusion = bool(int(config["lm"]["lm_fusion"]))
    
    if lm_fusion:
        lm_path = config["lm"]["lm_path"]
        lm_emb_size = int(config["lm"]["lm_emb_size"])
        lm_hidden_size = int(config["lm"]["lm_hidden_size"])
        lm_num_layers = int(config["lm"]["lm_num_layers"])

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    if debug:
        logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)  # to stdout
    else:
        logging.basicConfig(filename=log_path,
                            format="%(asctime)s %(message)s",
                            level=logging.DEBUG)

    logging.info("process id: {:d} is allocated".format(os.getpid()))
    logging.info("read config from {}".format(config_path))

    logging.info("model will saved to: {}".format(save_format))
    logging.info("optimizer will saved to: {}".format(optimizer_save_format))

    if lm_fusion:
        lm = RNNLM(vocab_size, lm_emb_size, lm_hidden_size, lm_num_layers)
        state_dict = torch.load(lm_path, map_location=device)
        lm.load_state_dict(state_dict)
        logging.info(f"load LM from {lm_path}")
        lm.eval()
    else:
        lm = None

    model = AttnModel(config_path, lm=lm)
    model.apply(init_weight)
    
    if args.model_path is not None:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"load model from {args.model_path}")

    model.to(device)

    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    if args.optim_path is not None:
        optim_state_dict = torch.load(args.optim_path, map_location=device)
        optimizer.load_state_dict(optim_state_dict)
        logging.info(f"load optimizer from {args.optim_path}")

    dataset = SpeechDataset(config_path)
    # for debugging
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_train,
                            num_workers=2, pin_memory=True)
    dataset_val = SpeechDataset(config_path, mode="valid")
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_train,
                                num_workers=2, pin_memory=True)

    num_steps = len(dataset)

    for epoch in range(args.st_epoch, num_epochs):
        loss_sum = 0

        for step, data in enumerate(dataloader):
            loss_step = train_step(model, optimizer, data, vocab_size, ls_type, ls_prob, distill_weight)
            loss_sum += loss_step

            if (step + 1) % log_step == 0:
                logging.info("epoch = {:>2} step = {:>6} / {:>6} loss = {:.3f}".format(epoch + 1,
                                                                                       (step + 1) * batch_size,
                                                                                       num_steps,
                                                                                       loss_sum / log_step))
                loss_sum = 0
                # DEBUG
                # break
        
        # get validation loss per epoch
        logging.info("start validation")
        val_loss = 0
        for data in dataloader_val:
            val_loss += val_step(model, data, vocab_size, ls_type, ls_prob, distill_weight)
        logging.info("epoch = {:>2} val_loss = {:.3f}".format(epoch + 1, val_loss))

        if epoch == 0 or (epoch + 1) % save_step == 0:
            save_path = save_format.format(epoch + 1)
            torch.save(model.state_dict(), save_path)
            optimizer_save_path = optimizer_save_format.format(epoch + 1)
            torch.save(optimizer.state_dict(), optimizer_save_path)
            logging.info("model saved to: {}".format(save_path))
            logging.info("optimizer saved to: {}".format(optimizer_save_path))
        update_epoch(model, epoch + 1)
        decay_lr(optimizer, epoch + 1, decay_start_epoch, decay_rate)

        # gradually increase distill_weight
        if distill_grad and (epoch + 1) >= distill_start_epoch:
            distill_weight = min(distill_weight_fin,
                                 distill_weight + (distill_weight_fin / distill_increase_range))
            logging.info(f"distill_weight is set to {distill_weight}")


if __name__ == "__main__":
    train()
