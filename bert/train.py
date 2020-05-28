import argparse
import configparser
import logging
import os
import random
import torch
from torch.utils.data import DataLoader
from pytorch_transformers import BertForMaskedLM, modeling_bert
from create_mask import create_masked_lm_labels
from optimization import BertAdam
import sys
sys.path.append("../")
from dataset import LMDataset


def train_dataset(dataset, model, optimizer, multi_gpu, device, epoch,
                  batch_size, num_steps, log_step, num_to_mask, mask_id, max_seq_len):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)
    loss_sum = 0
    loss_ds = 0

    cand_indices = [i for i in range(max_seq_len)]

    for step, data in enumerate(dataloader):
        x_batch = data[0].to(device)
        x_batch, masked_lm_labels = create_masked_lm_labels(x_batch,
                                                            max_seq_len,
                                                            cand_indices,
                                                            num_to_mask,
                                                            mask_id,
                                                            device)

        optimizer.zero_grad()

        loss, _ = model(x_batch, attention_mask=None, masked_lm_labels=masked_lm_labels)

        if multi_gpu:
            loss = loss.mean()

        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        loss_ds += loss.item()

        if (step + 1) % log_step == 0:
            logging.info(f"epoch = {epoch + 1} step {step + 1} / {num_steps}: {(loss_sum / log_step):.6f}")
            loss_sum = 0

    return loss_ds

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpu", type=str, default=None,
                        help="binary flag which gpu to use (For example '10100000' means use device_id=0 and 2)")

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.conf)

    hidden_size = int(config["model"]["hidden_size"])
    num_hidden_layers = int(config["model"]["num_hidden_layers"])
    num_attention_heads = int(config["model"]["num_attention_heads"])
    intermediate_size = int(config["model"]["intermediate_size"])
    max_position_embeddings = int(config["model"]["max_position_embeddings"])
    #
    vocab_size = int(config["vocab"]["vocab_size"])
    mask_id = int(config["vocab"]["mask_id"])
    #
    log_path = config["log"]["log_path"]
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)
    log_step = int(config["log"]["log_step"])
    #
    train_size = int(config["data"]["train_size"])
    #
    save_prefix = config["save"]["save_prefix"]
    save_dir = os.path.dirname(save_prefix)
    os.makedirs(save_dir, exist_ok=True)
    save_epoch = int(config["save"]["save_epoch"])
    #
    batch_size = int(config["train"]["batch_size"])
    if args.debug:
        batch_size = 10
    num_epochs = int(config["train"]["num_epochs"])
    learning_rate = float(config["train"]["learning_rate"])
    warmup_proportion = float(config["train"]["warmup_proportion"])
    weight_decay = float(config["train"]["weight_decay"])
    #
    num_to_mask = int(config["mask"]["num_to_mask"])
    max_seq_len = int(config["mask"]["max_seq_len"])

    if args.debug:
        logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(filename=log_path,
                            format="%(asctime)s %(message)s",
                            level=logging.DEBUG)

    bertconfig = modeling_bert.BertConfig(vocab_size_or_config_json_file=vocab_size,
                                          hidden_size=hidden_size,
                                          num_hidden_layers=num_hidden_layers,
                                          num_attention_heads=num_attention_heads,
                                          intermediate_size=intermediate_size,
                                          max_position_embeddings=max_position_embeddings)
    model = BertForMaskedLM(config=bertconfig)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.gpu is not None:
        device_ids = []
        for device_id, flag in enumerate(args.gpu):
            if flag == "1":
                device_ids.append(device_id)
        multi_gpu = True
        device = torch.device("cuda:{}".format(device_ids[0]))
    else:
        multi_gpu = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"device: {device}")
    if "model_path" in config["train"]:
        model_path = config["train"]["model_path"]
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"load model from {model_path}")
    model.to(device)
    if multi_gpu:
        logging.info(f"GPU: device_id={device_ids}")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.train()

    # optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = (train_size // batch_size) * num_epochs
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         weight_decay=weight_decay,
                         t_total=t_total)
    logging.info("start training...")

    for epoch in range(num_epochs):
        if "train_dir" in config["data"]:
            train_dir = config["data"]["train_dir"]
            datpaths = os.listdir(train_dir)
            random.shuffle(datpaths)
            for step_ds, path in enumerate(datpaths):
                path = os.path.join(train_dir, path)
                dataset = LMDataset(path)
                num_steps = (len(dataset) // batch_size) + 1
                logging.info(f"dataset from: {path}")
                loss_ds = train_dataset(dataset=dataset, 
                                        model=model,
                                        optimizer=optimizer,
                                        multi_gpu=multi_gpu,
                                        device=device,
                                        epoch=epoch,
                                        batch_size=batch_size,
                                        num_steps=num_steps,
                                        log_step=log_step,
                                        num_to_mask=num_to_mask,
                                        mask_id=mask_id,
                                        max_seq_len=max_seq_len)
                logging.info(f"step {step_ds + 1} / {len(datpaths)}: {(loss_ds / num_steps):.6f}")
        else:
            train_path = config["data"]["train_path"]
            dataset = LMDataset(train_path)
            num_steps = (len(dataset) // batch_size) + 1
            loss_epoch = train_dataset(dataset=dataset, 
                                       model=model,
                                       optimizer=optimizer,
                                       multi_gpu=multi_gpu,
                                       device=device,
                                       epoch=epoch,
                                       batch_size=batch_size,
                                       num_steps=num_steps,
                                       log_step=log_step,
                                       num_to_mask=num_to_mask,
                                       mask_id=mask_id,
                                       max_seq_len=max_seq_len)
            logging.info(f"epoch {epoch + 1} / {num_epochs} : {(loss_epoch / num_steps):.6f}")

        if (epoch + 1) % save_epoch == 0:
            save_path = f"{save_prefix}.network.epoch{(epoch + 1):d}"
            optimizer_save_path = f"{save_prefix}.optimizer.epoch{(epoch + 1):d}"
            if multi_gpu:
                torch.save(
                    model.module.state_dict(), save_path.format(epoch + 1)
                )
            else:
                torch.save(
                    model.state_dict(), save_path.format(epoch + 1)
                )
            logging.info(f"model saved: {save_path}")
            torch.save(
                optimizer.state_dict(), optimizer_save_path.format(epoch + 1)
            )
            logging.info(f"optimizer saved: {optimizer_save_path}")

if __name__ == "__main__":
    train()
