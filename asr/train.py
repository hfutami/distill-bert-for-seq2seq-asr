import argparse
import configparser
import logging
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from attnmodel.model import AttnModel
from dataset import SpeechDataset, collate_fn
from utils import distill_loss, label_smoothing_loss, init_weight, update_epoch, decay_lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(model, optimizer, data, vocab_size, ls_prob, distill_weight):
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
                            ls_prob)
    else:
        loss = label_smoothing_loss(preds,
                                    hard_labels,
                                    lab_lens,
                                    vocab_size,
                                    ls_prob)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    return loss.item()

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    debug = args.debug
    config = configparser.ConfigParser()
    config.read(args.conf)

    log_path = config["log"]["log_path"]
    log_step = int(config["log"]["log_step"])
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)

    save_prefix = config["save"]["save_prefix"]
    save_format = save_prefix + ".network.epoch{}"
    optimizer_save_format = save_prefix + ".optimizer.epoch{}"
    save_step = int(config["save"]["save_step"])
    save_dir = os.path.dirname(save_prefix)
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = int(config["train"]["num_epochs"])
    batch_size = int(config["train"]["batch_size"])
    decay_start_epoch = int(config["train"]["decay_start_epoch"])
    decay_rate = float(config["train"]["decay_rate"])
    vocab_size = int(config["vocab"]["vocab_size"])
    ls_prob = float(config["train"]["ls_prob"])
    distill_weight = float(config["distill"]["distill_weight"])

    if debug:
        logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)  # to stdout
    else:
        logging.basicConfig(filename=log_path,
                            format="%(asctime)s %(message)s",
                            level=logging.DEBUG)

    model = AttnModel(args.conf)
    model.apply(init_weight)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)

    dataset = SpeechDataset(args.conf)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=2, pin_memory=True)
    num_steps = len(dataloader)

    for epoch in range(num_epochs):
        loss_sum = 0

        for step, data in enumerate(dataloader):
            loss_step = train_step(model, optimizer, data, vocab_size, ls_prob, distill_weight)
            loss_sum += loss_step

            if (step + 1) % log_step == 0:
                logging.info("epoch = {:>2} step = {:>6} / {:>6} loss = {:.3f}".format(epoch + 1,
                                                                                       step + 1,
                                                                                       num_steps,
                                                                                       loss_sum / log_step))
                loss_sum = 0
        
        if epoch == 0 or (epoch + 1) % save_step == 0:
            save_path = save_format.format(epoch + 1)
            torch.save(model.state_dict(), save_path)
            optimizer_save_path = optimizer_save_format.format(epoch + 1)
            torch.save(optimizer.state_dict(), optimizer_save_path)
            logging.info("model saved to: {}".format(save_path))
            logging.info("optimizer saved to: {}".format(optimizer_save_path))
        update_epoch(model, epoch + 1)
        decay_lr(optimizer, epoch + 1, decay_start_epoch, decay_rate)

if __name__ == "__main__":
    train()
