import json
import re
import random
import preprocess
import paras
from Storage import Entity, Relation, BatchData
import matplotlib.pyplot as plt

class Data:
    def __init__(self, a, s):
        self.schemas = s
        self.text_raw = a['text']
        self.text = a['text'][:paras.MAX_LEN]
        self.text_len = len(self.text)
        if a['postag'] != []:
            self.postag = a['postag']
        else:
            self.postag = preprocess.get_pseg(self.text)
        self.poses = []
        for one in self.postag:
            self.poses.extend([one['pos']] * len(one['word']))
        self.poses = self.poses[:self.text_len]
        self.spo_list = []
        self.output = []
        if 'spo_list' in a:
            for one in a['spo_list']:
                type = self.schemas.predicate_to_id(one['subject_type'], one['predicate'], one['object_type'])
                if self.text.find(one['subject']) >= 0 and self.text.find(one['object']) >= 0:
                    tmp = {}
                    tmp['type'] = type
                    tmp['subject'] = one['subject']
                    tmp['object'] = one['object']
                    self.spo_list.append(tmp)

    def text_to_index(self, vocab):
        self.word_indexes = vocab.words_to_indexes(self.text)
        self.pos_indexes = vocab.pos_to_indexes(self.poses)
        assert len(self.word_indexes) == len(self.pos_indexes) == len(self.text)

    def set_label(self, type, start, length, value):
        for i in range(start, start+length):
            self.output[i][type] = value

    def get_output(self):
        if self.output == []:
            for i in range(self.text_len):
                self.output.append([0] * (paras.SCHEMA_NUMBER * 2))
            for one in self.spo_list:
                type = one['type']
                l = len(one['subject'])
                for i in range(self.text_len - l):
                    if self.text.startswith(one['subject']):
                        self.set_label(type*2, i, l, 1)
                l = len(one['object'])
                for i in range(self.text_len - l):
                    if self.text.startswith(one['object']):
                        self.set_label(type*2+1, i, l, 1)
        return self.output

    def output_result(self):
        res = {}
        res['text'] = self.text
        spo_list = []
        for one in self.spo_list:
            obj = self.schemas.output_result(one['type'])
            obj['subject'] = one['subject']
            obj['object'] = one['object']
            spo_list.append(obj)
        res['spo_list'] = spo_list
        return res

class Dataset:
    def __init__(self):
        self.data = []
        self.word_num = 0
        self.i = 0

    def add(self, d):
        self.data.append(d)
        self.word_num += len(d.text)

    def sort(self):
        self.data.sort(key=lambda x:x.text_len)

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
            a = json.loads(line.lower())
            b = Data(a, schemas)
            dataset.add(b)
    #dataset.sort()
    #hist = [one.text_len for one in dataset.data]
    #plt.hist(hist)
    #plt.show()
    return dataset