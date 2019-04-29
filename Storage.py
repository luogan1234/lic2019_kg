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
        self.entity_label = []
        self.subject_start = []
        self.subject_end = []
        self.object_start = []
        self.object_end = []
        self.relation_type = []

    def add(self, data):
        index = data.word_indexes + [0] * (paras.MAX_LEN - data.text_len)
        pos = data.pos_indexes + [0] * (paras.MAX_LEN - data.text_len)
        entity_label = data.get_step1_data()
        relation = data.get_step2_data()
        self.index.append(index)
        self.pos.append(pos)
        self.entity_label.append(entity_label)
        self.subject_start.append(relation['subject'].start)
        self.subject_end.append(relation['subject'].end)
        self.object_start.append(relation['object'].start)
        self.object_end.append(relation['object'].end)
        self.relation_type.append(relation['type'])