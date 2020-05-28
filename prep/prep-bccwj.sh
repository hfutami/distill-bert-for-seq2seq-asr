subword-nmt apply-bpe -c ../data/bpe/bpe2k.bpe --vocabulary ../data/bpe/bpe2k.vocab < ../data/bccwj/bccwj.lb > ../data/bccwj/bccwj.lb.bpe2k
subword-nmt apply-bpe -c ../data/bpe/bpe2k.bpe --vocabulary ../data/bpe/bpe2k.vocab < ../data/bccwj/bccwj.pb > ../data/bccwj/bccwj.pb.bpe2k
#
cut -d " " -f 1 ../data/bpe/bpe2k.vocab > ../data/bpe/bpe2k.vocab.word
cat ../data/bpe/vocab.special.bert ../data/bpe/bpe2k.vocab.word > ../data/bpe/vocab.bert
#
python text2id.py -text ../data/bccwj/bccwj.lb.bpe2k -vocab ../data/bpe/vocab.bert > ../data/bccwj/bccwj.lb.id.bert
python text2id.py -text ../data/bccwj/bccwj.pb.bpe2k -vocab ../data/bpe/vocab.bert > ../data/bccwj/bccwj.pb.id.bert
#
head -n 25 ../data/bccwj/bccwj.lb.id.bert > ../data/bccwj/bccwj.lb.id.bert.dev
tail -n +26 ../data/bccwj/bccwj.lb.id.bert > ../data/bccwj/bccwj.lb.id.bert.train
head -n 25 ../data/bccwj/bccwj.pb.id.bert > ../data/bccwj/bccwj.pb.id.bert.dev
tail -n +26 ../data/bccwj/bccwj.pb.id.bert > ../data/bccwj/bccwj.pb.id.bert.train
#
cat ../data/bccwj/bccwj.pb.id.bert.train ../data/bccwj/bccwj.lb.id.bert.train > ../data/bccwj/bccwj.id.bert.train
cat ../data/bccwj/bccwj.pb.id.bert.dev ../data/bccwj/bccwj.lb.id.bert.dev > ../data/bccwj/bccwj.id.bert.dev
#
python sample.py ../data/bccwj/bccwj.id.bert.train --seqlen 256 --shift 64 > ../data/bccwj/bccwj.id.bert.train.s256
python split.py ../data/bccwj/bccwj.id.bert.train.s256 ../data/bccwj/bccwj.s256/
