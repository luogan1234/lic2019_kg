import paras

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
        self.max_len = paras.MAX_LEN

    def add(self, data):
        self.index.append(data.word_indexes)
        self.pos.append(data.pos_indexes)
        self.output.append(data.get_output())
        self.max_len = max(self.max_len, data.text_len)

    def _padding(self, a):
        return a + [0] * (self.max_len - len(a))

    def padding(self):
        for i in range(len(self.index)):
            self.index[i] = self._padding(self.index[i])
            self.pos[i] = self._padding(self.pos[i])
            for j in range(self.max_len - len(self.output[i])):
                self.output[i].append([0] * (paras.SCHEMA_NUMBER * 2))