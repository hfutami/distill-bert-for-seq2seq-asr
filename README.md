## Distilling the Knowledge of BERT for Sequence-to-Sequence ASR

### Requirements

hoge

### Data preparation

We used two corpus: CSJ and BCCWJ.
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

### Pre-training BERT

### Training seq2seq ASR

baseline seq2seq ASR
```

```


### Result

|  | WER(eval1) |
|:---:|:---:|
| baseline | 10.31 |
| BERT(utterance) | 9.53 |
| BERT(full-length) | **9.19** |
