class Vocab:
    def __init__(self, vocab_path):
        with open(vocab_path) as f:
            lines = [line.strip() for line in f]
        
        i2w = []
        for line in lines:
            i2w.append(line)
        self.i2w = i2w

        w2i = {}
        for i, line in enumerate(lines):
            w2i[line] = i
        self.w2i = w2i

        self.unk_id = w2i["<unk>"]

    def id2word(self, i):
        return self.i2w[i]

    def ids2words(self, ids):
        return [self.id2word(i) for i in ids]

    def word2id(self, w):
        if w in self.w2i:
            return self.w2i[w]
        return self.unk_id
        
    def words2ids(self, words):
        return [self.word2id(w) for w in words]


def subword_to_word(subwords: list):
    tmp = ""
    words = []
    for subword in subwords:
        if subword[-2:] == "@@":
            tmp += subword[:-2]
        else:
            words.append(tmp + subword)
            tmp = ""
    return words
