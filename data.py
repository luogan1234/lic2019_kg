import json
import re
import random
import preprocess
import paras
from Storage import Entity, Relation, BatchData

class Data:
    def __init__(self, a, s):
        self.schemas = s
        self.text = a['text']
        if a['postag'] != []:
            self.postag = a['postag']
        else:
            self.postag = preprocess.get_pseg(self.text)
        self.poses = []
        for one in self.postag:
            self.poses.extend([one['pos']] * len(one['word']))
        self.text_len = len(self.text)
        self.spo_list = []
        self.entities = []
        self.relations = []
        self.entity_label = []
        if 'spo_list' in a:
            entity_label = []
            for i in range(len(self.schemas.types)):
                entity_label.append([0] * self.text_len)
            for one in a['spo_list']:
                one['subject_info'] = self.add_entity(entity_label, one['subject'], one['subject_type'])
                one['object_info'] = self.add_entity(entity_label, one['object'], one['object_type'])
            tmp_list = self.entity_label_to_entities(entity_label)
            for one in a['spo_list']:
                type = self.schemas.predicate_to_id(one['subject_type'], one['predicate'], one['object_type'])
                if type == 50:
                    print(one)
                a = one['subject_info']
                b = one['object_info']
                if a[0] >= 0 and b[0] >= 0:
                    sub = self.entities[tmp_list[a[2]][a[0]] - 1]
                    ob = self.entities[tmp_list[b[2]][b[0]] - 1]
                    self.relations.append(Relation(type, sub, ob))

    def entity_label_to_entities(self, entity_label):
        tmp_list = []
        self.entities = []
        for i in range(self.schemas.type_len):
            tmp_list.append([0] * self.text_len)
        num = 1
        for i in range(self.schemas.type_len):
            start = -1
            for j in range(self.text_len):
                if entity_label[i][j] == 1 and start == -1:
                    start = j
                if entity_label[i][j] == 1:
                    tmp_list[i][j] = num
                if entity_label[i][j] == 0 and start >= 0:
                    self.entities.append(Entity(start, j, i))
                    num += 1
                    start = -1
            if start >= 0:
                self.entities.append(Entity(start, self.text_len, i))
                num += 1
        return tmp_list

    def add_relation_result(self, relations):
        self.relations = relations

    def add_entity(self, entity_label, name, entity_type):
        name = re.sub('[《》]', '', name)
        start = self.text.find(name)
        end = start + len(name)
        type = -1
        if start >= 0:
            type = self.schemas.entity_to_id(entity_type)
            for i in range(start, end):
                entity_label[type][i] = 1
        return start, end, type

    def text_to_index(self, vocab):
        self.word_indexes = vocab.words_to_indexes(self.text)
        self.pos_indexes = vocab.pos_to_indexes(self.poses)
        assert len(self.word_indexes) == len(self.pos_indexes) == len(self.text)

    def get_step1_data(self):
        if self.entity_label == []:
            for i in range(paras.MAX_LEN):
                self.entity_label.append([0]*self.schemas.type_len)
            for entity in self.entities:
                for i in range(entity.start, entity.end):
                    self.entity_label[i][entity.type] = 1
        return self.entity_label

    def get_step2_data(self):
        m = len(self.relations)
        if m > 0 and self.spo_list == []:
            self.spo_list = []
            for relation in self.relations:
                obj = {}
                obj['subject'] = relation.subject
                obj['object'] = relation.object
                obj['type'] = relation.type
                self.spo_list.append(obj)
            '''
            tmp = []
            for e1 in self.entities:
                for e2 in self.entities:
                    if not any(e1 == relation.subject and e2 == relation.object for relation in self.relations):
                        obj = {}
                        obj['subject'] = e1
                        obj['object'] = e2
                        obj['type'] = 51
                        tmp.append(obj)
            random.shuffle(tmp)
            self.spo_list.extend(tmp[:m])
            '''
        x = random.randint(0, len(self.spo_list)-1)
        return self.spo_list[x]

    def output_result(self):
        res = {}
        res['text'] = self.text
        spo_list = []
        for relation in self.relations:
            obj = self.schemas.output_result(relation.type)
            obj['subject'] = self.text[relation.subject.start:relation.subject.end]
            obj['object'] = self.text[relation.object.start:relation.object.end]
            spo_list.append(obj)
        res['spo_list'] = spo_list
        return res

class Dataset:
    def __init__(self):
        self.data = []
        self.phrases = set()
        self.word_num = 0
        self.i = 0

    def add(self, d):
        self.data.append(d)
        self.word_num += len(d.text)
        for one in d.postag:
            self.phrases.add(one['word'])

    def get_indexes(self, vocab):
        for one in self.data:
            one.text_to_index(vocab)

    def next_batch(self, batch_size):
        batch_data = BatchData()
        n = len(self.data)
        while batch_size > 0:
            data = self.data[self.i]
            self.i += 1
            if data.relations == []:
                continue
            batch_data.add(data)
            batch_size -= 1
            if self.i >= n:
                self.i -= n
        return batch_data

    def output_result(self):
        res = []
        for one in self.data:
            res.append(json.dumps(one.output_result(), ensure_ascii=False))
        return '\n'.join(res)

    def print_info(self):
        print('data number:', len(self.data))
        print('word number:', self.word_num)
        print('distinct phrase number:', len(self.phrases))

def load_data(file, schemas):
    dataset = Dataset()
    with open(file, 'r', encoding='utf-8') as f:
        tmp = f.read().split('\n')
        for line in tmp:
            if line == '':
                continue
            a = json.loads(line.lower())
            b = Data(a, schemas)
            dataset.add(b)
    return dataset