class Vocab:
    def __init__(self):
        self.word_to_index = {'<pad>': 0}
        self.index_to_word = ['<pad>']
        self.pos_to_index = {'<pad>': 0}
        self.index_to_pos = ['<pad>']
        self.num_word = 1
        self.num_pos = 1
        self.max_len = 0

    def words_to_indexes(self, sen):
        res = []
        for word in sen:
            if word not in self.word_to_index:
                self.word_to_index[word] = self.num_word
                self.index_to_word.append(word)
                self.num_word += 1
            res.append(self.word_to_index[word])
        self.max_len = max(self.max_len, len(res))
        return res

    def pos_to_indexes(self, sen):
        res = []
        for pos in sen:
            if pos not in self.pos_to_index:
                self.pos_to_index[pos] = self.num_pos
                self.index_to_pos.append(pos)
                self.num_pos += 1
            res.append(self.pos_to_index[pos])
        return res

    def print_info(self):
        print('distinct words:', self.num_word, 'distinct pos:', self.num_pos, 'max len:', self.max_len)