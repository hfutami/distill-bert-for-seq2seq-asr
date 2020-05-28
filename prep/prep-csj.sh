cut -d " " -f 2- ../data/csj/csj.aps > ../data/csj/csj.aps.text
cut -d " " -f 1 ../data/csj/csj.aps > ../data/csj/csj.aps.path
cut -d " " -f 2- ../data/csj/csj.sps > ../data/csj/csj.sps.text
cat ../data/csj/csj.aps.text ../data/csj/csj.sps.text > ../data/csj/csj.text
#
subword-nmt apply-bpe -c ../data/bpe/bpe2k.bpe --vocabulary ../data/bpe/bpe2k.vocab < ../data/csj/csj.text > ../data/csj/csj.text.bpe2k
subword-nmt apply-bpe -c ../data/bpe/bpe2k.bpe --vocabulary ../data/bpe/bpe2k.vocab < ../data/csj/csj.aps.text > ../data/csj/csj.aps.text.bpe2k
#
cut -d " " -f 1 ../data/bpe/bpe2k.vocab > ../data/bpe/bpe2k.vocab.word
cat ../data/bpe/vocab.special.bert ../data/bpe/bpe2k.vocab.word > ../data/bpe/vocab.bert
#
python text2id.py -text ../data/csj/csj.text.bpe2k -vocab ../data/bpe/vocab.bert > ../data/csj/csj.id.bert
python text2id.py -text ../data/csj/csj.aps.text.bpe2k -vocab ../data/bpe/vocab.bert > ../data/csj/csj.aps.id.bert
# for pre-train BERT
python sample.py ../data/csj/csj.id.bert --seqlen 256 --shift 64 > ../data/csj/csj.id.bert.train.s256
# for soft label preparation
paste -d " " ../data/csj/csj.aps.path ../data/csj/csj.aps.id.bert > ../data/csj/csj.aps.pathaid.bert
python addbictx.py ../data/csj/csj.aps.pathaid.bert -num_ctx 256 > ../data/csj/csj.aps.id.bert.full
python mask.py ../data/csj/csj.aps.pathaid.bert --no_ctx > ../data/csj/csj.aps.pathaid.bert.utt.masked
python mask.py ../data/csj/csj.aps.id.bert.full > ../data/csj/csj.aps.pathaid.bert.full.masked
# for train seq2seq ASR
cat ../data/bpe/vocab.special.asr ../data/bpe/bpe2k.vocab.word > ../data/bpe/vocab.asr
python text2id.py -text ../data/csj/csj.aps.text.bpe2k -vocab ../data/bpe/vocab.asr --addsp > ../data/csj/csj.aps.id.asr
paste -d " " ../data/csj/csj.aps.path ../data/csj/csj.aps.id.asr > ../data/csj/csj.aps.pathaid.asr
python sort_xlen.py ../data/csj/csj.aps.pathaid.asr > ../data/csj/csj.aps.pathaid.asr.train
