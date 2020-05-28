## Distilling the Knowledge of BERT for Sequence-to-Sequence ASR

### Requirements
subword-nmt https://github.com/rsennrich/subword-nmt
pytorch
transformers https://github.com/huggingface/transformers

### Data preparation

We used two corpus:
the Corpus of Spontaneous Japanese (CSJ) and the Balanced Corpus of Contemporary Written Japanese (BCCWJ).
CSJ is for training of ASR and BERT, and BCCWJ is for training of BERT.

1. Prepare CSJ-APS and CSJ-SPS data in the same format as `./data/csj.example`.
They should be put as `./data/csj.aps` and `./data/csj.sps`.

2. Prepare BCCWJ-LB and BCCWJ-PB data in the same format as `./data/bccwj.example`.
They should be put as `./data/bccwj.lb` and `./data/bccwj.pb`.

3. run `./prep-bccwj.sh` (cd: `./prep`)

4. run `./prep-csj.sh` (cd: `./prep`)

### Pre-train BERT

We used TITAN X (12GB) x3 for pre-training.

```
(cd: ./bert)
python train.py -conf bert-bccwj.config
python train.py -conf bert-csj.config
```

### Soft label preparation
```
(cd: ./bert)
python prep_soft_labels.py -conf bert-csj.config -model checkpoints/bert.csj.network.epoch50 -ctx utt
python prep_soft_labels.py -conf bert-csj.config -model checkpoints/bert.csj.network.epoch50 -ctx full
```

### Train seq2seq ASR

baseline seq2seq ASR
```
(cd: ./asr)
python train.py -conf base.config
```
seq2seq ASR with soft labels from BERT (utterance-level)
```
(cd: ./asr)
python train.py -conf bert-utt.config
```
seq2seq ASR with soft labels from BERT (full-length)
```
(cd: ./asr)
python train.py -conf bert-full.config
```

### Test ASR

```
(cd: ./asr)
python test.py -conf {base.config or bert-utt.config or bert-full.config}
```

### Result

|  | WER(eval1) |
|:---:|:---:|
| baseline | 10.31 |
| BERT(utterance) | 9.53 |
| BERT(full-length) | **9.19** |
