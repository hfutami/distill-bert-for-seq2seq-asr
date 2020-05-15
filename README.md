## Distilling the Knowledge of BERT for Sequence-to-Sequence ASR

### Requirements
subword-nmt https://github.com/rsennrich/subword-nmt

### Data preparation

We used two corpus:
the Corpus of Spontaneous Japanese (CSJ) and the Balanced Corpus of Contemporary Written Japanese (BCCWJ).
CSJ is for training of ASR and BERT, and BCCWJ is for training of BERT.

BCCWJ
```
cd ./prep/bccwj
bash prep.sh {PATH_TO_YOUR_BCCWJ}
```

CSJ
```
cd ./prep/csj
bash prep.sh {PATH_TO_YOUR_CSJ}
```

### Pre-train BERT


### Soft label preparation

```
```

### Train seq2seq ASR

baseline seq2seq ASR
```
python train.py -conf base.config
```
seq2seq ASR with soft labels from BERT (utterance-level)
```
python train.py -conf bert_utt.config
```
seq2seq ASR with soft labels from BERT (full-length)
```
python train.py -conf bert_full.config
```

### Test ASR

```
python test.py -conf {base.config or bert_utt.config or bert_full.config}
```

### Result

|  | WER(eval1) |
|:---:|:---:|
| baseline | 10.31 |
| BERT(utterance) | 9.53 |
| BERT(full-length) | **9.19** |
