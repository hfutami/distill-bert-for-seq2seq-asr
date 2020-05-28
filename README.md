## Distilling the Knowledge of BERT for Sequence-to-Sequence ASR

### Requirements
subword-nmt https://github.com/rsennrich/subword-nmt

### Data preparation

We used two corpus:
the Corpus of Spontaneous Japanese (CSJ) and the Balanced Corpus of Contemporary Written Japanese (BCCWJ).
CSJ is for training of ASR and BERT, and BCCWJ is for training of BERT.

1. Prepare CSJ-APS and BCCWJ-SPS data in the same format as `./data/csj.example`.
They should be put as `./data/csj.aps` and `./data/csj.sps`.

2. Prepare BCCWJ-LB and BCCWJ-PB data in the same format as `./data/bccwj.example`.
They should be put as `./data/bccwj.lb` and `./data/bccwj.pb`.

3. run `./prep-csj.sh` (at `./prep`)
4. run `./prep-bccwj.sh` (at `./prep`)

### Pre-train BERT
```
(at ./bert)
python train.py -conf bert-bccwj.config
python train.py -conf bert-csj.config
```

### Soft label preparation
```
(at ./bert)
python prep_soft_labels.py -model checkpoints/bert.csj.epoch50
```

### Train seq2seq ASR

baseline seq2seq ASR
```
(at ./asr)
python train.py -conf base.config
```
seq2seq ASR with soft labels from BERT (utterance-level)
```
(at ./asr)
python train.py -conf bert-utt.config
```
seq2seq ASR with soft labels from BERT (full-length)
```
(at ./asr)
python train.py -conf bert-full.config
```

### Test ASR

```
python test.py -conf {base.config or bert-utt.config or bert-full.config}
```

### Result

|  | WER(eval1) |
|:---:|:---:|
| baseline | 10.31 |
| BERT(utterance) | 9.53 |
| BERT(full-length) | **9.19** |
