import json
import re
import random
import preprocess
import paras
import numpy as np
from Storage import Entity, Relation, BatchData
import matplotlib.pyplot as plt
import time

tot1 = [0]*50
tot2 = [0]*25
class Data:
    def __init__(self, a, s):
        global tot1, tot2
        self.schemas = s
        self.text_raw = a['text']
        if a['postag'] != []:
            self.postag = a['postag']
        else:
            self.postag = preprocess.get_pseg(self.text_raw)
        self.text = []
        self.poses = []
        for one in self.postag:
            self.poses.append(one['pos'])
            self.text.append(one['word'].lower())
        self.text = self.text[:paras.MAX_LEN]
        self.poses = self.poses[:paras.MAX_LEN]
        self.text_len = len(self.text)
        self.spo_list = []
        self.output = []
        self.relation_data = []
        if 'spo_list' in a:
            for one in a['spo_list']:
                r_type = self.schemas.predicate_to_id(one['predicate'])
                s_type = self.schemas.entity_to_id(one['subject_type'])
                o_type = self.schemas.entity_to_id(one['object_type'])
                tot1[r_type] += 1
                tot2[s_type] += 1
                tot2[o_type] += 1
                if self.find(one['subject'].lower()) != [] and self.find(one['object'].lower()) != []:
                    tmp = {}
                    tmp['r_type'] = r_type
                    tmp['s_type'] = s_type
                    tmp['o_type'] = o_type
                    tmp['subject'] = one['subject'].lower()
                    tmp['object'] = one['object'].lower()
                    self.spo_list.append(tmp)
    
    def find(self, target):
        global tot
        pos = []
        for i in range(0, self.text_len):
            phrase = self.text[i]
            j = i + 1
            while target.startswith(phrase):
                if target == phrase:
                    pos.append([i, j])
                if j >= self.text_len:
                    break
                phrase += self.text[j]
                j += 1
        return pos

    def text_to_index(self, vocab):
        self.word_indexes = vocab.words_to_indexes(self.text)
        self.pos_indexes = vocab.pos_to_indexes(self.poses)
        assert len(self.word_indexes) == len(self.pos_indexes) == len(self.text)

    def set_label(self, type, start, end):
        for i in range(start, end-1):
            if self.output[i] == 0:
                self.output[i] = 1
        if self.output[start] in [3, 4]:
            self.output[start] = 4
        else:
            self.output[start] = 2
        if self.output[end-1] in [2, 4]:
            self.output[end-1] = 4
        else:
            self.output[end-1] = 3

    def get_output(self):
        if self.output == []:
            for i in range(self.text_len):
                self.output.append(0)
            for one in self.spo_list:
                pos = self.find(one['subject'])
                for l, r in pos:
                    self.set_label(one['s_type'], l, r)
                pos = self.find(one['object'])
                for l, r in pos:
                    self.set_label(one['o_type'], l, r)
        return self.output
    
    def get_relation_data(self):
        if self.relation_data == []:
            for one in self.spo_list:
                pos = self.find(one['subject'])
                s_l = pos[0][0]
                s_r = pos[0][1]
                pos = self.find(one['object'])
                o_l = pos[0][0]
                o_r = pos[0][1]
                m_l = min(s_r, o_r)
                m_r = max(s_l, o_l, m_l)
                left = min(s_l, o_l) - 1
                right = max(s_r, o_r)
                d = m_r - m_l
                r_type = one['r_type']
                self.relation_data.append([s_l, s_r, o_l, o_r, m_l, m_r, left, right, d, r_type])
            if self.relation_data == []:
                self.relation_data.append([0, 0, 0, 0, 0 ,0, 0, 0, 0, paras.SCHEMA_NUMBER - 1])
        return self.relation_data
    
    def label_to_entity(self, j, outputs):
        entities = []
        start = -1
        for i in range(self.text_len):
            if outputs[i][j] == 1 and start == -1:
                start = i
            if start >=0 and outputs[i][j] == 0:
                if i > start + 1:
                    entities.append([start, i])
                start = -1
        if start >= 0 and self.text_len > start + 1:
            entities.append([start, self.text_len])
        return entities
    
    def output_to_spo_list(self, outputs):
        spo_list = []
        filter = set()
        for i in range(len(outputs)):
            if outputs[i] not in range(49):
                continue
            r = self.relation_data[i]
            sub_name = self.text_raw[r[0]:r[1]]
            ob_name = self.text_raw[r[2]:r[3]]
            r_type = r[-1]
            merge = sub_name + '#' + ob_name
            if merge not in filter:
                filter.add(merge)
                spo_list.append({'r_type': r_type, 'subject': sub_name, 'object': ob_name})
        self.spo_list = spo_list

    def output_result(self):
        res = {}
        res['text'] = self.text_raw
        spo_list = []
        for one in self.spo_list:
            obj = self.schemas.output_result(one['r_type'])
            obj['subject'] = one['subject']
            obj['object'] = one['object']
            spo_list.append(obj)
        res['spo_list'] = spo_list
        return res

class Dataset:
    def __init__(self):
        self.data = []
        self.word_num = 0
        self.num = 0
    
    def init_batch(self, shuffle=True):
        if shuffle:
            random.shuffle(self.data)
        self.i = 0

    def add(self, d):
        self.data.append(d)
        self.word_num += len(d.text)
        self.num += 1

    def sort(self):
        self.data.sort(key=lambda x: x.text_len)

    def get_indexes(self, vocab):
        for one in self.data:
            one.text_to_index(vocab)

    def next_batch(self, batch_size):
        batch_data = BatchData()
        n = len(self.data)
        while batch_size > 0:
            data = self.data[self.i]
            self.i += 1
            batch_size -= 1
            batch_data.add(data)
            if self.i >= n:
                self.i -= n
        batch_data.padding()
        return batch_data

    def output_result(self):
        res = []
        for one in self.data:
            res.append(json.dumps(one.output_result(), ensure_ascii=False))
        return '\n'.join(res)

    def print_info(self):
        print('data number:', len(self.data))
        print('word number:', self.word_num)

def load_data(file, schemas):
    dataset = Dataset()
    with open(file, 'r', encoding='utf-8') as f:
        tmp = f.read().split('\n')
        for line in tmp:
            if line == '':
                continue
            a = json.loads(line)
            b = Data(a, schemas)
            if 'spo_list' not in a or len(b.spo_list) == len(a['spo_list']):
                dataset.add(b)
    #hist = [one.text_len for one in dataset.data]
    #plt.hist(hist)
    #plt.show()
    return dataset