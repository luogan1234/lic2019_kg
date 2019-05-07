import paras
import random

class Entity:
    def __init__(self, start, end, type):
        self.start = start
        self.end = end
        self.type = type

class Relation:
    def __init__(self, type, subject, object):
        self.type = type
        self.subject = subject
        self.object = object

class BatchData:
    def __init__(self):
        self.index = []
        self.pos = []
        self.output = []
        self.relation_data = []
        self.seq_length = []
        self.max_len = 0
        self.batch_size = 0

    def add(self, data):
        self.index.append(data.word_indexes)
        self.pos.append(data.pos_indexes)
        self.output.append(data.get_output())
        relation_data = data.get_relation_data()
        p = random.randint(0, len(relation_data) - 1)
        choose = relation_data[p].copy()
        p2 = random.randint(1, paras.SCHEMA_NUMBER)
        if p2 == paras.SCHEMA_NUMBER:
            choose = choose[2:4] + choose[:2] + choose[4:-1] + [paras.NONE_SCHEMA]
        self.relation_data.append(choose)
        self.seq_length.append(data.text_len)
        self.max_len = max(self.max_len, data.text_len)
        self.batch_size += 1

    def _padding(self, a):
        return a + [0] * (self.max_len - len(a))

    def padding(self):
        for i in range(len(self.index)):
            self.index[i] = self._padding(self.index[i])
            self.pos[i] = self._padding(self.pos[i])
            for j in range(self.max_len - len(self.output[i])):
                self.output[i].append(0)