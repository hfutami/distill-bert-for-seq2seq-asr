from random import shuffle, sample
import numpy as np
import torch

def create_masked_lm_labels(x_batch, max_seq_len, cand_indices, num_to_mask, mask_id, device):
    masked_lm_labels_batch = []

    for i, x in enumerate(x_batch):
        shuffle(cand_indices)
        mask_indices = sorted(sample(cand_indices, num_to_mask))
        masked_lm_labels = np.full(max_seq_len, dtype=np.int, fill_value=-1)

        for index in mask_indices:
            # everytime, replace with [MASK]
            masked_token = mask_id
            masked_lm_labels[index] = x[index]
            x_batch[i, index] = masked_token

        masked_lm_labels_batch.append(masked_lm_labels)

    return x_batch, torch.tensor(masked_lm_labels_batch).to(device)
