cut -d " " -f 2- ../data/csj/csj.aps > ../data/csj/csj.aps.text
cut -d " " -f 2- ../data/csj/csj.sps > ../data/csj/csj.sps.text
cat ../data/csj/csj.aps.text ../data/csj/csj.sps.text > ../data/csj/csj.text
#
subword-nmt apply-bpe -c ../data/bpe/bpe2k.bpe --vocabulary ../data/bpe/bpe2k.vocab < ../data/csj/csj.text > ../data/csj/csj.text.bpe2k
subword-nmt apply-bpe -c ../data/bpe/bpe2k.bpe --vocabulary ../data/bpe/bpe2k.vocab < ../data/csj/csj.text > ../data/csj/csj.aps.text.bpe2k
#
cut -d " " -f 1 ../data/bpe/bpe2k.vocab > ../data/bccwj/bpe2k.vocab.word
cat ../data/bpe/vocab.special.bert ../data/bpe2k.vocab.word > ../data/vocab.bert
#
python text2id.py -text ../data/csj/csj.text.bpe2k -vocab ../data/bpe/vocab.bert > ../data/csj/csj.id.bert
# for pre-train BERT
python sample.py ../data/csj/csj.id.bert --seqlen 256 --shift 64 > ../data/csj/csj.id.bert.train.s256
# for soft label preparation

# for train seq2seq ASR
