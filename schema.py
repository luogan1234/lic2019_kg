import json
import paras

merge_dict = {'历史人物': '人物', '国家': '地点', '行政区': '地点', '图书作品': '书籍', '企业': '机构'}
class Schema:
    def __init__(self, subject_type, predicate, object_type):
        self.subject_type_raw = subject_type
        if subject_type in merge_dict:
            subject_type = merge_dict[subject_type]
        self.object_type_raw = object_type
        if object_type in merge_dict:
            object_type = merge_dict[object_type]
        self.subject_type = subject_type
        self.predicate = predicate
        self.object_type = object_type
    
    def match(self, subject_type, object_type):
        return self.subject_type == subject_type and self.object_type == object_type

    def match2(self, predicate):
        return self.predicate == predicate

    def output_result(self):
        res = {'subject_type': self.subject_type_raw, 'predicate': self.predicate, 'object_type': self.object_type_raw}
        return res

    @staticmethod
    def default_result():
        res = {'subject_type': '', 'predicate': '', 'object_type': ''}
        return res

class Schemas:
    def __init__(self):
        self.schemas = []
        self.types = []
        self.type_len = 0
    
    def find_matched_schema_ids(self, subject_type, object_type):
        res = []
        for i, schema in enumerate(self.schemas):
            if schema.match(subject_type, object_type):
                res.append(i)
        return res

    def predicate_to_id(self, predicate):
        for i, schema in enumerate(self.schemas):
            if schema.match2(predicate):
                return i
        raise ValueError
    
    def entity_to_id(self, entity):
        if entity in merge_dict:
            entity = merge_dict[entity]
        if entity in self.types:
            return self.types.index(entity)
        raise ValueError

    def add_schema(self, s):
        self.schemas.append(s)
        if s.subject_type not in self.types:
            self.types.append(s.subject_type)
        if s.object_type not in self.types:
            self.types.append(s.object_type)
        self.type_len = len(self.types)

    def output_result(self, index):
        if index in range(50):
            return self.schemas[index].output_result()
        else:
            return Schema.default_result()

    def print_info(self):
        print('schema number:', len(self.schemas))
        print('entity types:', len(self.types))
        print(self.types)

def load_schema():
    s = Schemas()
    with open(paras.SCHEMAS, 'r', encoding='utf-8') as f:
        tmp = f.read().split('\n')
        for line in tmp:
            if line == '':
                continue
            a = json.loads(line)
            b = Schema(a['subject_type'], a['predicate'], a['object_type'])
            s.add_schema(b)
    return s

if __name__ == '__main__':
    s = load_schema()
    s.print_info()
    #for schema in s.schemas:
    #    print(s.find_matched_schema_ids(schema.subject_type, schema.object_type))